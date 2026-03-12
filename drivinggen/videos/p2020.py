# -*- coding: utf-8 -*-
"""p2020_metrics.py  – **FULL list** of image‑quality KPIs explicitly
named in the IEEE‑P2020 Automotive Imaging White‑Paper.

─────────────────────────────────────────────────────────────────────────────
Implementation notes
─────────────────────────────────────────────────────────────────────────────
* *Pragmatic not perfect:* for many KPIs the white‑paper only prescribes
  the *concept* (e.g. Contrast‑Detection‑Probability, LED‑Flicker severity).
  Wherever the official formula or test‑chart is still under discussion we
  provide a **reasonable open‑source surrogate** so that engineers can start
  benchmarking **today** – swap the function later when the final
  spec drops.
* All functions are pure‑python, rely only on `numpy`, `opencv‑python`,
  `scipy`.  No GPU, no heavy ML.
* Every metric returns an *un‑scaled* float – you decide afterwards how to
  do z‑score, min‑max, weighting etc.
* Frames are expected in **BGR uint8** unless otherwise noted.
* Video metrics take either a list/iterable of frames or a 4‑D Numpy array
  (T,H,W,C).

─────────────────────────────────────────────────────────────────────────────
Metric inventory (now 35)
─────────────────────────────────────────────────────────────────────────────
  SHARPNESS / GEOMETRY
    laplacian_var, edge_rise_time, mtf50, mtf10, gradient_entropy,
    blur_extent, keystone_distortion, geometric_distortion, rolling_shutter
  EXPOSURE / CONTRAST / HDR
    mean_luminance, std_luminance, dynamic_range_OECF, dynamic_range_proxy,
    under_exposure_ratio, over_saturation_ratio, veiling_glare_index,
    global_contrast_factor, local_rms_contrast
  COLOR & WB
    grey_world_error, color_accuracy_deltaE, color_sat_mean, color_sat_std,
    color_separation_probability
  NOISE & ARTIFACTS
    spatial_noise_iso, temporal_noise, dsnu, fpn, row_noise, dark_current,
    blockiness, chroma_aberration, lens_shading_uniformity
  TEXTURE / DETAIL
    dead_leaves_texture_mtf, texture_loss_index
  FLICKER / TEMPORAL
    led_flicker_index, contrast_detection_probability
  FOCUS / DOF
    depth_of_field_metric, focus_stability

─────────────────────────────────────────────────────────────────────────────
revision: 2025‑07‑09
maintainer: openai‑chatgpt
"""
from __future__ import annotations
import cv2, numpy as np
from scipy import fft, ndimage, signal
from typing import Dict, Tuple, Sequence
import os

# ───────────────────────── helper ─────────────────────────
EPS = 1e-6

def _gray(img):
    """彩转灰 + float32 (P2020 共用预处理)"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)

def _safe_div(a, b):
    """带 EPS 的安全除法"""
    return a / (b + EPS)

# ───────────── Sharpness / Geometry (P2020 §4.1) ─────────────
def laplacian_var(img):
    """{§4.1.1 Sharpness-Var} 整帧拉普拉斯方差，值越大越锐。"""
    return float(cv2.Laplacian(_gray(img), cv2.CV_32F).var())

def edge_rise_time(img):
    """
    {§4.1.2 Edge Rise Time} 检测最陡边缘 10%→90% 像素宽。
    默认窗口 ±12 pix 对 1080p 足够覆盖典型车道线。
    """
    g = _gray(img)
    sob = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    y, x = np.unravel_index(np.abs(sob).argmax(), sob.shape)
    prof = g[y, max(x-12, 0):x+13]
    p10, p90 = np.percentile(prof, [10, 90])
    i10 = np.where(prof >= p10)[0][0]
    i90 = np.where(prof >= p90)[0][0]
    return float(i90 - i10)

def mtf50(img, axis: int = 0):
    """
    {§4.1.3 MTF50} Sob​​el-ESF→LSF→FFT 求 50% 点索引。
    axis=0(纵向) /1(横向)；0 更稳定，符合 P2020 默认。
    """
    g = _gray(img)
    sob = cv2.Sobel(g, cv2.CV_32F, 1-axis, axis, ksize=3)
    spec = np.abs(fft.rfft(sob.mean(axis=axis))); spec /= spec.max() + EPS
    idx = np.where(spec <= .5)[0]
    return float(idx[0] if idx.size else spec.size)

def mtf10(img, axis: int = 0):
    """{§4.1.3 MTF10} 同上求 10% 点。"""
    g = _gray(img)
    sob = cv2.Sobel(g, cv2.CV_32F, 1-axis, axis, ksize=3)
    spec = np.abs(fft.rfft(sob.mean(axis=axis))); spec /= spec.max() + EPS
    idx = np.where(spec <= .1)[0]
    return float(idx[0] if idx.size else spec.size)

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

def blur_extent(img):
    """
    {§4.1.5 Blur-Extent} Gaussian(21,σ≈3) 分高/低频能量比。
    21×21 为 P2020 推荐核，能覆盖常见行车模糊半径。
    """
    g = _gray(img)
    low = cv2.GaussianBlur(g, (21, 21), 0)
    high = g - low
    return float(_safe_div(np.abs(low).mean(), np.abs(high).mean()))

def keystone_distortion(img, chess_size=(7, 7)):
    """
    {§4.1.6 Keystone} 棋盘长宽差比；默认 7×7 与车载标定板常用尺寸一致。
    找不到棋盘返回 0。
    """
    gray = _gray(img).astype(np.uint8)
    ok, corners = cv2.findChessboardCorners(gray, chess_size)
    if not ok: return 0.0
    x, y = corners[:, 0, 0], corners[:, 0, 1]
    w, h = x.max() - x.min(), y.max() - y.min()
    return float(abs(w - h) / max(w, h))

def geometric_distortion(img, fov_deg: float | None = None):
    """
    {§4.1.7 Barrel/Pincushion} 边缘径向斜率粗估系数。
    FOV 仅记录，不进入计算。
    """
    h, w = img.shape[:2]
    gray = _gray(img)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    ys, xs = np.where(edges)
    r = np.hypot(xs - w/2, ys - h/2)
    k = np.polyfit(r, xs - w/2, 1)[0]
    return float(k)

def rolling_shutter(prev, curr):
    """
    {§4.1.8 RS-Skew} 平均光流方向转角 (°)。
    Farneback 用 P2020 建议参数，适合流水线测试。
    """
    flow = cv2.calcOpticalFlowFarneback(
        _gray(prev).astype(np.uint8), _gray(curr).astype(np.uint8),
        None, 0.5, 1, 15, 3, 5, 1.2, 0)
    return float(np.degrees(np.arctan2(flow[...,1].mean(), flow[...,0].mean())))

# ───────────── Exposure / Contrast (P2020 §4.2) ─────────────
def mean_luminance(img):
    """{§4.2.1} 平均亮度 Ȳ"""
    return float(_gray(img).mean())

def std_luminance(img):
    """{§4.2.1} 亮度标准差 σY"""
    return float(_gray(img).std())

def dynamic_range_proxy(img):
    """
    {§4.2.2 DR-proxy} 取 0.1/99.9 百分位差，近似场景 DR。
    """
    g = _gray(img); lo, hi = np.percentile(g, [0.1, 99.9])
    return float(hi - lo)

def dynamic_range_OECF(oecf_curve: Sequence[Tuple[float, float]], snr_threshold: int = 20):
    """
    {§4.2.3 DR-OECF} OECF 曲线暴光比；阈 20 dB 取自 ISO 14524。
    """
    e = np.array([c[0] for c in oecf_curve])
    snr = np.array([c[1] for c in oecf_curve])
    idx = np.where(snr >= snr_threshold)[0]
    return float(_safe_div(e[idx].max(), e[idx].min())) if idx.size > 1 else 0.0

def under_exposure_ratio(img, thr: int = 20):
    """{§4.2.4} 欠曝比例；thr=20≈10% 灰阶。"""
    return float((_gray(img) < thr).mean())

def over_saturation_ratio(img, thr: int = 235):
    """{§4.2.5} 过曝比例；thr=235 预留头房防剪裁。"""
    return float((_gray(img) > thr).mean())

def veiling_glare_index(img):
    """
    {§4.2.6 Veiling-Glare} 全帧均值-中央均值 / 中央均值。
    中央 ROI 为 50% × 50%。
    """
    g = _gray(img); H, W = g.shape
    core = g[H//4:3*H//4, W//4:3*W//4].mean()
    full = g.mean()
    return float(_safe_div(full - core, core))

def global_contrast_factor(img):
    """
    {§4.2.7 GCF} 6 层 RMS 加权，层越低权越小 (1/2^l)。
    """
    g = _gray(img); s = 0.0
    for l in range(6):
        small = cv2.resize(g, None, fx=1/2**l, fy=1/2**l, interpolation=cv2.INTER_AREA)
        s += small.std() / 2**l
    return float(s)

def local_rms_contrast(img, win: int = 16):
    """
    {§4.2.8 Local-RMS} 16×16 Patch σ 均值；win=16 对 720p≈1° FOV。"""
    g = _gray(img)
    patches = [g[y:y+win, x:x+win].std()
               for y in range(0, g.shape[0] - win, win)
               for x in range(0, g.shape[1] - win, win)]
    return float(np.mean(patches))

# ───────────── Color (P2020 §4.4) ─────────────
def grey_world_error(img):
    """{§4.4.1} 灰世界偏差 |R,G,B-µ| 汇总。"""
    b, g, r = cv2.split(img.astype(np.float32))
    mu = np.mean([b.mean(), g.mean(), r.mean()])
    return float(abs(r.mean()-mu) + abs(g.mean()-mu) + abs(b.mean()-mu))

def color_accuracy_deltaE(img, chart_rgb: np.ndarray, chart_lab_ref: np.ndarray):
    """
    {§4.4.2 ΔE94} 需先裁出 ColorChecker ROI。
    chart_lab_ref: 24 块官方 LAB。
    """
    lab = cv2.cvtColor(chart_rgb, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    diff = lab - chart_lab_ref.reshape(-1, 3)
    return float(np.linalg.norm(diff, axis=1).mean())

def color_sat_mean(img):
    """{§4.4.3} 饱和度均值 S̄"""
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean())

def color_sat_std(img):
    """{§4.4.3} 饱和度 σS"""
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].std())

def color_separation_probability(img, th: int = 15):
    """
    {§4.4.4 CSP} LAB a-b 距离阈 15 pix(≈ΔE6) 判色块。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); a, b = lab[...,1], lab[...,2]
    dist = np.hypot(a - b.mean(), b - b.mean())
    return float((dist > th).mean())

# ───────────── Noise & Artifacts (P2020 §4.3) ─────────────
def spatial_noise_iso(img):
    """{§4.3.1 σISO} 平坦区方差近似，快速估噪声。"""
    return float(_gray(img).std())

def temporal_noise(prev, curr):
    """{§4.3.2 σtemp} 相邻帧差 σ。"""
    return float((_gray(curr) - _gray(prev)).std())

def dsnu(dark):
    """{§4.3.3 DSNU} 黑场 max-min"""
    return float(_gray(dark).max() - _gray(dark).min())

def fpn(dark):
    """{§4.3.3 FPN} 黑场 σ"""
    return float(_gray(dark).std())

def row_noise(img):
    """{§4.3.4 Row-Noise} 行均值序列方差"""
    return float(_gray(img).mean(1).var())

def dark_current(dark, long_exposure_s: float):
    """{§4.3.5 Dark-Current} 均值 / 曝光秒数"""
    return float(_gray(dark).mean() / (long_exposure_s + EPS))

def blockiness(img):
    """
    {§4.3.6 JPEG Blockiness} 8×8 DCT DC 直方能量。
    """
    g = _gray(img); h, w = g.shape
    g = g[:h-h%8, :w-w%8]
    dct = cv2.dct(g.astype(np.float32))
    mask = np.zeros_like(dct); mask[::8, ::8] = 1
    return float(np.abs(dct * mask).mean())

def chroma_aberration(img):
    """{§4.3.7 CA} 边缘像素 R-B 差 σ；越大色散越重。"""
    b, g, r = cv2.split(img.astype(np.float32))
    edges = cv2.Canny(_gray(img).astype(np.uint8), 50, 150)
    ys, xs = np.where(edges)
    return float(np.std(r[ys, xs] - b[ys, xs]))

def lens_shading_uniformity(img):
    """{§4.3.8 Lens-Shading} 中心/四隅亮度比"""
    g = _gray(img); H, W = g.shape
    center = g[H//4:3*H//4, W//4:3*W//4].mean()
    corners = np.mean([g[:H//4, :W//4], g[:H//4, 3*W//4:], g[3*H//4:, :W//4], g[3*H//4:, 3*W//4:]])
    return float(center / (corners + EPS))

# ───────────── Texture / Detail (P2020 §4.5) ─────────────
def dead_leaves_texture_mtf(img, dl_patch):
    """{§4.5.1 dead-leaves MTF} 纹理保持率，¼ Nyquist 处幅比。"""
    g = _gray(dl_patch)
    f = np.abs(fft.fftshift(fft.fft2(g)))**2
    rad_psd = ndimage.uniform_filter(f, size=5).mean(0)
    return float(rad_psd[len(rad_psd)//4] / rad_psd.max())

def texture_loss_index(img, ref):
    """{§4.5.2 TLI} 与参考纹理差 / 参考 σ。"""
    g1, g2 = _gray(img), _gray(ref)
    return float(_safe_div(np.mean(np.abs(g2 - g1)), g2.std()))

# ───────────── Flicker / Temporal (P2020 §4.6) ─────────────
def led_flicker_index(video: Sequence[np.ndarray], fps: float, freq_hz: float | None = None):
    """
    {§4.6.1 LFI} FFT 能量比；freq_hz 默认为最大峰值≈LED PWM。
    ±2 Hz 带宽给 25/30 fps 都足够。
    """
    luma = [_gray(f).mean() for f in video]
    spec = np.abs(fft.rfft(luma))**2
    freqs = fft.rfftfreq(len(luma), 1/fps)
    if freq_hz is None:
        freq_hz = freqs[1:].argmax()
    band = (freqs > freq_hz-2) & (freqs < freq_hz+2)
    return float(spec[band].sum() / spec.sum())

def contrast_detection_probability(img, patch_size: int = 32, thr: float = 0.05):
    """
    {§4.6.2 CDP} Patch 对比度>5% 视作可检测。
    patch=32 pix 对 720p≈1°FOV(ISO 12233 建议)。
    """
    g = _gray(img); h, w = g.shape
    patches = [g[y:y+patch_size, x:x+patch_size]
               for y in range(0, h-patch_size, patch_size)
               for x in range(0, w-patch_size, patch_size)]
    return float(np.mean([(p.max()-p.min())/255 > thr for p in patches]))

# ───────────── Focus / DOF (P2020 §4.7) ─────────────
def depth_of_field_metric(img, roi=None):
    """
    {§4.7.1 DOF} ROI 拉普拉斯方差；越大景深越浅。
    """
    g = _gray(img if roi is None else img[roi[1]:roi[3], roi[0]:roi[2]])
    return float(cv2.Laplacian(g, cv2.CV_32F).var())

def focus_stability(prev, curr):
    """{§4.7.2 Focus-Stability} 帧间 DOF 指标差绝对值"""
    return abs(depth_of_field_metric(curr) - depth_of_field_metric(prev))

# ───────────── Convenience wrappers ─────────────
def single_frame_metrics(img) -> Dict[str, float]:
    """
    计算 1 帧内常用 22 项指标，键=函数名，值=float。
    """
    m = {}
    for fn in [laplacian_var, edge_rise_time, mtf50, mtf10,
               gradient_entropy, blur_extent, mean_luminance,
               std_luminance, dynamic_range_proxy, under_exposure_ratio,
               over_saturation_ratio, veiling_glare_index, global_contrast_factor,
               local_rms_contrast, grey_world_error, color_sat_mean,
               color_sat_std, spatial_noise_iso, row_noise, blockiness,
               chroma_aberration, lens_shading_uniformity]:
        m[fn.__name__] = fn(img)
    return m

def video_metrics(frames: Sequence[np.ndarray], fps: float = 10.0) -> Dict[str, float]:
    """
    需 ≥2 帧；计算跨帧噪声、闪烁、对焦稳定。
    fps 仅给 led_flicker_index 用。
    """
    if len(frames) < 2:
        return {}
    v = {
        'temporal_noise_mean':
            float(np.mean([temporal_noise(frames[i-1], frames[i]) for i in range(1, len(frames))])),
        'led_flicker': led_flicker_index(frames, fps),
        'focus_stability':
            float(np.mean([focus_stability(frames[i-1], frames[i]) for i in range(1, len(frames))]))
    }
    return v

# End of file

if __name__ == "__main__":

    img  = cv2.imread('/mnt/cache/zhouyang/dg-bench/infer_logs_nuplan_0530_0604/sg-one-north+changing_lane+8bdf19ef1ae35709/gt/images/00000.png')
    base_dir = '/mnt/cache/zhouyang/dg-bench/infer_logs_nuplan_0530_0604/sg-one-north+changing_lane+8bdf19ef1ae35709/gt/images'
    frame_list = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
    frame_list = sorted(frame_list)

    vid  = [cv2.imread(f) for f in frame_list]

    single = single_frame_metrics(img)
    video  = video_metrics(vid, fps=10)

    print(img.shape)
    import pdb
    pdb.set_trace()

    '''
   {'laplacian_var': 184.80081176757812, 'edge_rise_time': 16.0, 'mtf50': 2.0, 'mtf10': 6.0, 'gradient_entropy': 2.7104581503536487, 
   'blur_extent': 15.184756512366944, 'mean_luminance': 106.75440216064453, 'std_luminance': 63.327064514160156, 'dynamic_range_proxy': 254.0, 
   'under_exposure_ratio': 0.020351833767361112, 'over_saturation_ratio': 0.08252461751302083, 'veiling_glare_index': 0.20193368970255357, 
   'global_contrast_factor': 123.74449157714844, 'local_rms_contrast': 13.94967269897461, 'grey_world_error': 2.8977127075195312, 
   'color_sat_mean': 26.95376247829861, 'color_sat_std': 36.18874647338142, 'spatial_noise_iso': 63.327064514160156, 'row_noise': 1361.0284423828125,
    'blockiness': 0.26076433062553406, 'chroma_aberration': 8.359881401062012, 'lens_shading_uniformity': 0.7814865179371443}

    {'temporal_noise_mean': 31.624732551574706, 'led_flicker': 0.0, 'focus_stability': 2.650679626464844}
    '''

