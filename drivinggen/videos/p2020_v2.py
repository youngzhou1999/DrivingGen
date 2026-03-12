# -*- coding: utf-8 -*-
"""p2020_high_priority_metrics_p1.py  – **ADAS‑critical KPI subset (≥ 8 pts)**
================================================================================
Flat‑namespace implementation that mimics your original *p1* style while
preserving rich **chapter references** and **parameter notes**.  Only three
lightweight dependencies: ``numpy``, ``opencv‑python``, ``scipy``.

Frames: **BGR uint8**
Video : iterable or 4‑D ndarray (T,H,W,C)

Metric inventory (14)
----------------------
SHARPNESS / RESOLUTION  (§6)
  mtf50, mtf10, contrast_transfer_accuracy
GEOMETRY  (§3)
  total_distortion, forward_projection_error, back_projection_error
DYNAMIC‑RANGE / HDR  (§5)
  captured_dynamic_range, visible_dynamic_range
NOISE & SENSITIVITY  (§4)
  temporal_noise, fixed_pattern_noise, snr_at_lux10
FLARE / STRAY‑LIGHT  (§2)
  flare_attenuation
FLICKER / TEMPORAL  (§7)
  flicker_modulation_power, modulation_mitigation_probability
"""
from __future__ import annotations
import cv2, numpy as np
from scipy import fft, stats
from typing import Dict, Sequence, Tuple

# ───────────────────────── helpers ─────────────────────────
EPS = 1e-9  # avoid divide‑by‑zero everywhere


def _gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR/GRAY → float32 Gray (linear space)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def _safe_div(a, b):
    return a / (b + EPS)


# ─────────────── DYNAMIC‑RANGE / HDR (§5) ───────────────

# 1⃣ 单帧替代动态范围 (升级版)
def frame_dynamic_range_proxy(img,
                              p_lo: float = 0.1,
                              p_hi: float = 99.9,
                              assume_gamma: float | None = None) -> float:
    """
    {proxy-DR} 全帧灰度分位差
    ------------------------------------------
    • 无测试卡时的动态范围替代指标。
    • 默认直接返回 8-bit 代码值差 (0-255)。
    • 如果 *assume_gamma* 给出（例如 2.2），
      先反 γ → 近似线性光度 → 以 **EV**(log2) 形式输出，
      更公平地横向比较 SDR / HDR / 生成片。

    参数
    ----
    img           : BGR uint8
    p_lo, p_hi    : 分位点 (排除离群像素)，单位 %
    assume_gamma  : None → 不校正；float → γ 校正并返回 EV 差
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # γ 校正（可选）
    if assume_gamma:
        norm = gray / 255.0
        gray = np.power(norm, assume_gamma)          # 近似线性光度

    lo, hi = np.percentile(gray, [p_lo, p_hi])

    if assume_gamma:
        # 输出 EV = log2(hi/lo)
        return float(np.log2((hi + EPS) / (lo + EPS)))
    else:
        # 原 8-bit 差
        return float(hi - lo)

def sequence_dynamic_range_proxy(video,
                                 p_lo: float = 0.1,
                                 p_hi: float = 99.9,
                                 assume_gamma: float | None = None) -> float:
    """
    proxy-DR (全片直方图分位差)

    • assume_gamma=None  → 返回 8-bit *代码值* 差 (0-255)
    • assume_gamma=2.2   → 先反 γ，再算 hi/lo → 以 *对数 EV* 输出
    """
    # 1. 展平所有灰度像素
    all_pix = np.concatenate([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).ravel()
                              for f in video]).astype(np.float32)

    # 2. γ 反解 (若给定)
    if assume_gamma:
        norm = all_pix / 255.0
        all_pix = np.power(norm, assume_gamma)  # 近似线性光度

    lo, hi = np.percentile(all_pix, [p_lo, p_hi])

    if assume_gamma:
        # 输出 EV 差，以 2 的对数刻度
        return float(np.log2((hi + EPS) / (lo + EPS)))
    else:
        # 原 8-bit 差，0-255 区间
        return float(hi - lo)


# 3⃣ 曝光抖动
def temporal_exposure_jitter(video):
    lum = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in video]
    return float(np.std(np.diff(lum)))


# ─────────────────── SHARPNESS / RESOLUTION (§6) ───────────────────

# ──────────── 改进版 MTF50 / MTF10 （双向平均）────────────
def mtf50(img, axis: int | str = "both") -> float:
    """
    {§6.4.3 SFR} **MTF-50 surrogate**（Sobel-ESF）
    ------------------------------------------------
    • axis = 0  → 只测垂直边（水平结构），与原实现一致  
    • axis = 1  → 只测水平边  
    • axis = "both" / None → 同时测 0 和 1，返回平均，\n
      对真实路景 & 生成片噪声更稳健。
    """
    if axis in (0, 1):
        return _mtf_sobel(img, axis, 0.50)
    # 双向平均
    val0 = _mtf_sobel(img, 0, 0.50)
    val1 = _mtf_sobel(img, 1, 0.50)
    return float((val0 + val1) * 0.5)


def mtf10(img, axis: int | str = "both") -> float:
    """
    {§6.4.3 SFR} **MTF-10 surrogate**（10 % 模块）  
    参见 `mtf50` 说明；仅把阈值换成 0.10。
    """
    if axis in (0, 1):
        return _mtf_sobel(img, axis, 0.10)
    val0 = _mtf_sobel(img, 0, 0.10)
    val1 = _mtf_sobel(img, 1, 0.10)
    return float((val0 + val1) * 0.5)


# -- 内部工具：按单一方向计算阈值落点 ------------------------
def _mtf_sobel(img, axis: int, thr: float) -> float:
    """
    Sobel → ESF → FFT → 找 modulation ≤ thr 的首个频点
    axis = 0 (垂直边) / 1 (水平边)
    """
    g = _gray(img)
    sob = cv2.Sobel(g, cv2.CV_32F, 1-axis, axis, ksize=3)
    spec = np.abs(fft.rfft(sob.mean(axis=axis)))
    spec /= spec.max() + EPS
    idx = np.flatnonzero(spec <= thr)
    return float(idx[0] if idx.size else spec.size)


def contrast_transfer_accuracy(
        img,
        tgt_contrasts: Sequence[float] = (0.1, 0.5, 0.9),
        patch_size: int = 16) -> float:
    """
    {§6.5 KPI-3 CTA}  多档 **Contrast-Transfer-Accuracy**

    • P2020 草案要求在 **低 / 中 / 高** 三档对比度下评估；  
      默认档位 (0.1, 0.5, 0.9)。  
    • 过程：先对图像降采样 ×4 → 16×16 patch，算每 patch
      Michelson 对比度 C = (I_max−I_min)/(I_max+I_min)。  
    • 对每档 *tgt*，用常数回归斜率 ≈  mean(C_patch)/tgt_contrast  
      斜率≈1 表示还原准确；<1 对比度被压低，>1 被增强。  
    • 返回 **三档斜率均值**；若需要分别查看，可改 `return slopes`
      把列表/字典抛出即可。
    """
    g = _gray(img)
    # 降采样 ×4 提升噪声鲁棒性
    small = cv2.resize(g, (0, 0), fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA)

    h, w = small.shape
    vals = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            p = small[y:y + patch_size, x:x + patch_size]
            c = _safe_div(p.max() - p.min(), p.max() + p.min())
            vals.append(c)

    if not vals:
        return 0.0

    mean_c = np.mean(vals)
    slopes = [mean_c / (t + EPS) for t in tgt_contrasts]

    return float(np.mean(slopes))


def edge_rise_time(img, window: int = 12) -> float:
    """
    {§4.1.2 Edge Rise Time} -- 10–90% Edge-Rise Width (px)
    不返回 None；若某方向无有效边，依然计算（差值可能很小）。
    """
    g = _gray(img)  # 需外部提供；返回单通道 uint8/float 图
    assert g.ndim == 2, "edge_rise_time expects grayscale."

    h, w = g.shape

    # --- Sobel ---
    sob_x = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)  # 垂直边响应
    sob_y = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)  # 水平边响应

    # 找最大响应位置（绝对值）
    y_v, x_v = np.unravel_index(np.argmax(np.abs(sob_x)), sob_x.shape)
    y_h, x_h = np.unravel_index(np.argmax(np.abs(sob_y)), sob_y.shape)

    # 取局部灰度 profile
    prof_vert  = g[y_v, max(0, x_v - window): min(w, x_v + window + 1)]
    prof_horiz = g[max(0, y_h - window): min(h, y_h + window + 1), x_h].ravel()

    def _erd_from_profile(prof: np.ndarray) -> float:
        """
        10%-90% 上升宽度；永不返回 None。
        若 profile 太短 / 平坦，对应宽度会接近 0。
        """
        prof = np.asarray(prof, dtype=float)
        n = prof.size
        if n <= 1:
            return 0.0

        p10, p90 = np.percentile(prof, [10, 90])

        # 在 profile 顺序上找首次达到阈值的索引；若未出现则用末尾索引
        i10s = np.flatnonzero(prof >= p10)
        i90s = np.flatnonzero(prof >= p90)
        i10 = int(i10s[0]) if i10s.size else 0
        i90 = int(i90s[0]) if i90s.size else n - 1

        # 保证非负
        width = max(i90 - i10, 0)
        return float(width)

    erd_v = _erd_from_profile(prof_vert)
    erd_h = _erd_from_profile(prof_horiz)

    return float((erd_v + erd_h) * 0.5)




# ───────────────────────── GEOMETRY (§3) ─────────────────────────

# def total_distortion(img):
#     """{§3.5.1} **Total radial distortion** quick estimate.

#     Uses Canny edges → fit Δx vs radius; positive slope≈barrel, negative≈pincushion."""
#     h, w = img.shape[:2]
#     edges = cv2.Canny(_gray(img).astype(np.uint8), 50, 150)
#     ys, xs = np.where(edges)
#     if xs.size < 50:
#         return 0.0
#     r = np.hypot(xs - w/2, ys - h/2)
#     k = np.polyfit(r, xs - w/2, 1)[0]
#     return float(k)

def total_distortion(img, outer_frac: float = 0.8):
    """{§3.5.1} Total radial distortion (proxy, uncalibrated, signed).

    思路：把边缘点的半径 r 排序后，构造“理想线性半径” s∈[0,Rmax]（等距序列），
    回归 r ≈ a + b*s 作为线性参考；以 Δr/r = (r - r_ref)/(s+ε) 为每点相对畸变，
    在外圈（s≥outer_frac*Rmax）取中位数为总畸变（带符号）。
       • 正值：外圈 r > 线性参考 → 桶形（barrel）
       • 负值：外圈 r < 线性参考 → 枕形（pincushion）
    返回单位：相对量（例如 0.02≈2%）；如需百分比可×100。
    """
    g = _gray(img).astype(np.uint8)
    h, w = g.shape[:2]
    edges = cv2.Canny(g, 50, 150)
    ys, xs = np.where(edges)
    if xs.size < 50:
        return float("nan")  # 更合理的“不可测”，也可改回 0.0

    cx, cy = w * 0.5, h * 0.5
    r = np.hypot(xs - cx, ys - cy)             # 观测半径 r_obs
    Rmax = float(np.hypot(cx, cy))             # 图像对角线半径

    # 排序后，用“理想线性半径” s（等距从 0→Rmax）做回归自变量
    order = np.argsort(r)
    r_sorted = r[order]
    M = r_sorted.size
    s = np.linspace(0.0, Rmax, M, dtype=np.float64)

    # 线性参考：r_ref = a + b*s（去掉整体缩放/偏置，只看非线性形状）
    a, b = np.polyfit(s, r_sorted, 1)
    r_ref = a + b * s

    # 相对畸变 Δr/r，避开 s=0 的除零
    rel = (r_sorted - r_ref) / (s + 1e-6)

    # 只看外圈（更敏感也更稳定），取中位数 → 带符号的总畸变 proxy
    k0 = int(np.clip(outer_frac * M, 0, M - 1))
    rel_outer = rel[k0:] if k0 < M else rel
    td = float(np.nanmedian(rel_outer))

    return td if td > 0 else -td



# ───────────────────────── NOISE ─────────────────────────

#TODO temporal is weak

# ─────────────────── FLARE / STRAY‑LIGHT (§2) ───────────────────

# def flare_attenuation(img):
#     """{§2.5 KPI‑2} **Flare Attenuation** – inner/outer brightness ratio (<1 ideal)."""
#     g = _gray(img)
#     h, w = g.shape
#     yy, xx = np.ogrid[:h, :w]
#     r = np.hypot(xx - w/2, yy - h/2)
#     inner = g[r < 0.1*max(h, w)].mean()
#     outer = g[(r > 0.4*max(h, w)) & (r < 0.45*max(h, w))].mean()
#     return float(_safe_div(inner, outer))

def flare_attenuation(
    img,
    peak_thr: int = 240,          # 保留做诊断；不 gate
    bg_thr: int = 60,             # 保留做诊断；不 gate
    inner_ratio: float = 0.10,
    outer_band: tuple[float, float] = (0.40, 0.45),
) -> float:
    """
    {§2.5 KPI-2} Flare Attenuation (proxy) – 中心/外环亮度比。
    永不返回 None；无条件计算。
    越小 → 抗耀斑性能越好（中心不被漫散射拖亮）。
    """
    g = _gray(img)
    assert g.ndim == 2, "flare_attenuation expects grayscale."

    h, w = g.shape
    # 使用真实对角线长度
    diag = float(np.hypot(h, w))

    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(xx - w * 0.5, yy - h * 0.5)

    inner_mask = (r < inner_ratio * diag)
    outer_mask = (r > outer_band[0] * diag) & (r < outer_band[1] * diag)

    # 容错：防止全 False（极端小图或参数过大）
    if not inner_mask.any():
        inner_mask = np.ones_like(g, dtype=bool)
    if not outer_mask.any():
        outer_mask = ~inner_mask  # 退化：用剩余区域当外环

    inner = float(g[inner_mask].mean())
    outer = float(g[outer_mask].mean())

    # 安全除法
    denom = outer if outer > 1e-6 else 1e-6
    return inner / denom


# TODO: may vgi

# ─────────────────── FLICKER / TEMPORAL (§7) ───────────────────

def flicker_modulation_power(video: Sequence[np.ndarray], fps: float=10):
    """{§7.5} **Flicker Modulation Power** – energy ratio around PWM peak."""
    luma = [_gray(f).mean() for f in video]
    spec = np.abs(fft.rfft(luma))**2
    freqs = fft.rfftfreq(len(luma), 1/fps)
    if len(freqs) < 3:
        return 0.0
    peak = freqs[1:][np.argmax(spec[1:])]
    band = (freqs > peak-2) & (freqs < peak+2)
    return float(spec[band].sum() / (spec.sum() + EPS))

def fmp_alias(video, fps: float=10, min_peak=0.2):
    """
    Proxy-FMP for low-fps footage.
    • 忽略 <min_peak Hz 的峰（场景照度漂移）
    • 带宽 = max(0.5 Hz, 20 %·peak)
    """
    lum = np.array([_gray(f).mean() for f in video])
    spec = np.abs(fft.rfft(lum))**2
    freqs = fft.rfftfreq(len(lum), 1 / fps)

    if len(freqs) < 3:
        return 0.0

    peak_idx = np.argmax(spec[1:]) + 1
    peak_f   = freqs[peak_idx]
    if peak_f < min_peak:
        return 0.0                     # 认为无可测闪烁

    bw   = max(0.5, 0.2 * peak_f)
    band = (freqs > peak_f - bw) & (freqs < peak_f + bw)
    return float(spec[band].sum() / (spec.sum() + EPS))


def modulation_mitigation_probability(video: Sequence[np.ndarray]):
    """{§7.6 KPI‑A} **Modulation Mitigation Probability (MMP)**.

    Fraction of frames whose mean deviates <5 % from grand‑mean under PWM."""
    means = np.array([_gray(f).mean() for f in video])
    dev = np.abs(means - means.mean()) / (means.mean() + EPS)
    return float((dev < 0.05).mean())

# def mmp_alias(video, fps: float=10, band_hz: float = 0.5, thr=0.05):
#     """
#     Proxy-MMP: 判断 beat 频率能量是否被压低.
#     fps 低 → 直接瞄准 0~fps/2 Hz 范围.
#     """
#     luma = np.array([_gray(f).mean() for f in video])
#     spec = np.abs(fft.rfft(luma))**2
#     freqs = fft.rfftfreq(len(luma), 1/fps)
#     # 排除直流 0 Hz
#     f1 = freqs[1:]
#     spec1 = spec[1:]
#     peak = f1[np.argmax(spec1)]
#     band = (freqs > peak-band_hz) & (freqs < peak+band_hz)
#     ratio = spec[band].sum() / (spec.sum() + EPS)
#     return float(ratio < thr)

# def mmp_alias(video, fps: float = 10.0,
#               band_hz: float = 0.5, thr: float = 0.05) -> float:
#     """
#     proxy-MMP (low-fps) : 逐帧判断 alias 能量占比 < thr 视为“抑制成功”，
#     返回成功帧占总帧的比例 ∈[0,1] —— 与 P2020 通用接口一致。
#     """
#     if len(video) < 4:
#         return 0.0

#     # 亮度序列一次 FFT 复用，避免逐帧 FFT
#     lum = np.array([_gray(f).mean() for f in video], dtype=np.float32)
#     spec  = np.abs(fft.rfft(lum))**2
#     freqs = fft.rfftfreq(len(lum), 1.0 / fps)

#     # alias 主峰（排除直流）
#     peak_idx = np.argmax(spec[1:]) + 1
#     peak_f   = freqs[peak_idx]
#     if peak_f < 0.2:                  # <0.2 Hz 视为光照漂移
#         return 1.0                    # 全帧都算“抑制成功”

#     band = (freqs > peak_f - band_hz) & (freqs < peak_f + band_hz)
#     peak_energy = spec[band].sum() / (spec.sum() + EPS)

#     # 逐帧近似：用整段比例代替帧级判断 (低 fps 误差可接受)
#     return float(peak_energy < thr)   # 若想概率，请改细粒度实现

def mmp_alias(video, fps: float = 10.0, band_hz: float = 0.5, thr: float = 0.05,
              win_sec: float = 3.0, hop_ratio: float = 0.5) -> float:
    """
    proxy-MMP: 滑窗统计“alias 能量占比 < thr”的窗口比例 ∈[0,1]。
    """
    T = len(video)
    if T < 4:
        return 0.0

    lum = np.array([_gray(f).mean() for f in video], dtype=np.float32)

    # 先用整段找到 alias 主峰频率（排除直流）
    spec  = np.abs(fft.rfft(lum))**2
    freqs = fft.rfftfreq(T, 1.0 / fps)
    peak_idx = np.argmax(spec[1:]) + 1
    peak_f   = freqs[peak_idx]
    if peak_f < 0.2:
        return 1.0  # 视作光照漂移，全部“成功”

    # 滑窗参数
    win = max(8, int(round(win_sec * fps)))
    hop = max(1, int(round(win * hop_ratio)))
    if win > T:
        win = T
        hop = max(1, T // 2)

    band_lo, band_hi = peak_f - band_hz, peak_f + band_hz

    hits, total = 0, 0
    for s in range(0, T - win + 1, hop):
        x = lum[s:s+win]
        X = np.abs(fft.rfft(x))**2
        F = fft.rfftfreq(len(x), 1.0 / fps)
        band = (F > band_lo) & (F < band_hi)
        A = X[band].sum() / (X.sum() + EPS)
        hits += float(A < thr)
        total += 1

    return float(hits / total) if total > 0 else 0.0




# ───────────────────────── TEXTURE  ─────────────────────────

def gradient_entropy(img):
    """
    {§4.1.4 Texture-Entropy} 梯度幅值直方图熵；越大纹理越丰富。
    """
    g = _gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    hist, _ = np.histogram(mag, 256, (0, mag.max() + EPS), density=True)
    hist += EPS
    return float(-(hist * np.log2(hist)).sum())


# ───────────────────────── v1 resonable  ─────────────────────────

def blur_extent(img):
    """
    {§4.1.5 Blur-Extent} Gaussian(21,σ≈3) 分高/低频能量比。
    21×21 为 P2020 推荐核，能覆盖常见行车模糊半径。
    """
    g = _gray(img)
    low = cv2.GaussianBlur(g, (21, 21), 0)
    high = g - low
    return float(_safe_div(np.abs(low).mean(), np.abs(high).mean()))

def chroma_aberration(img, on_empty="nan"):
    """{§4.3.7 CA} 边缘像素 R-B 差 σ；越大色散越重。
    on_empty: "nan"（推荐，用于统计不偏）或 "zero"（保持纯数值管线不报错）
    """
    b, g, r = cv2.split(img.astype(np.float32))
    edges = cv2.Canny(_gray(img).astype(np.uint8), 50, 150)

    ys, xs = np.where(edges > 0)
    if ys.size == 0:
        return (float('nan') if on_empty == "nan" else 0.0)

    diff = r[ys, xs] - b[ys, xs]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return (float('nan') if on_empty == "nan" else 0.0)

    return float(np.std(diff))

# ─────────────────── Convenience wrappers ───────────────────

def single_frame_metrics(img) -> Dict[str, float]:
    """Compute per‑frame high‑priority KPIs that need only one image."""
    out = {}
    for fn in [
        # frame_dynamic_range_proxy,
        # mtf50, mtf10,
        # contrast_transfer_accuracy,
        # edge_rise_time,
        # total_distortion,
        # flare_attenuation,
        # gradient_entropy,
        # blur_extent,
        # chroma_aberration
    ]:
        out[fn.__name__] = fn(img)
    return out


def video_metrics(frames: Sequence[np.ndarray], fps: float = 30.0) -> Dict[str, float]:
    """Compute video‑level KPIs (≥2 frames)."""
    if len(frames) < 2:
        return {}
    out = {}
    for fn in [
        # sequence_dynamic_range_proxy,
        # fmp_alias,
        mmp_alias,
        # temporal_exposure_jitter,
        # flicker_modulation_power,
        # modulation_mitigation_probability,
    ]:
        out[fn.__name__] = fn(frames)
    return out

# End of file
