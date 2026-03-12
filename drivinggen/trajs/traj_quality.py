import numpy as np
from typing import Tuple, Union, Iterable
ArrayLike = Union[np.ndarray, Iterable]

def _prep_xy(arr: np.ndarray, axis: int):
    arr = arr[..., :2]
    """把时间轴挪到 -2 并展平成 (N,T,2)，返回展平数组和原 batch 形状。"""
    arr = np.moveaxis(arr, axis, -2)          # (..., T, 2)
    batch_shape = arr.shape[:-2]
    T = arr.shape[-2]
    arr = arr.reshape(-1, T, 2)               # (N, T, 2)
    return arr, batch_shape

def comfort_score_norm(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,
    axis: int = -2,
    eps: float = 1e-9,
    v_static: float = 0.1,
    pct: float = None,          # 时间维先取百分位(如 95)；None=取均值
    length_eps: float = 1.0,    # 最小里程门限；≤该值视为无效
    j_scale: float = 1.0,       # 三项缩放(可按需调节量纲)
    a_scale: float = 1.0,
    y_scale: float = 1.0,
    reduce: str = "none",       # "none" → 返回每条；否则返回数据集均值
    return_components: bool = False,  # True→同时返回三项子分数
):
    """
    S_comf ∈ (0,1]，越大越好（与 curvature 一致：S = 1/(1 + value) 型映射）
    - 先得到三项“单位里程尖峰”：jerk_per_m, acc_per_m, yaw_per_m
    - 分别做 S_j = 1/(1 + (jerk_per_m/j_scale)) 等
    - 几何平均得到总分：S_comf = (S_j * S_a * S_y)^(1/3)
    - 从未移动 or 里程过短 → NaN（与 curvature 保持一致）
    """
    xy = np.asarray(traj_xy, float)
    # 统一到 (N,T,2)
    if axis != -2:
        xy = np.moveaxis(xy, axis, -2)
    bs = xy.shape[:-2]
    N, T, _ = xy.reshape(-1, xy.shape[-2], 2).shape
    xy_s = xy.reshape(-1, xy.shape[-2], 2)

    if T < 5:
        raise ValueError("Need ≥5 frames")

    # 速度/加速度/jerk（中心差分）
    v = (xy_s[:, 2:, :] - xy_s[:, :-2, :]) / (2 * dt)                      # (N,T-2,2)
    a = (xy_s[:, 2:, :] - 2*xy_s[:, 1:-1, :] + xy_s[:, :-2, :]) / (dt**2)  # (N,T-2,2)
    a_c = a[:, 1:-1, :]                                                    # (N,T-4,2)
    j = (a[:, 2:, :] - a[:, :-2, :]) / (2 * dt)                            # (N,T-4,2)

    # 速度模长与“是否移动”
    speed = np.linalg.norm(v, axis=-1)                                     # (N,T-2)
    moving = speed.max(axis=1) >= v_static                                 # (N,)

    # yaw-rate（由速度方向差分）
    th2 = np.arctan2(v[:, 2:, 1], v[:, 2:, 0])
    th0 = np.arctan2(v[:, :-2, 1], v[:, :-2, 0])
    yaw_rt = ((th2 - th0 + np.pi) % (2*np.pi) - np.pi) / (2 * dt)          # (N,T-4)

    # 三项时间聚合：均值或百分位
    def t_reduce(x):
        return np.percentile(x, pct, axis=1) if pct is not None else x.mean(axis=1)

    jerk_p = t_reduce(np.linalg.norm(j, axis=-1))            # (N,)
    acc_p  = t_reduce(np.linalg.norm(a_c, axis=-1))          # (N,)
    yaw_p  = t_reduce(np.abs(yaw_rt))                        # (N,)

    # 里程（与 v_static 判定口径一致）
    lengths = np.sum(np.linalg.norm(np.diff(xy_s, axis=1), axis=-1), axis=1)  # (N,)
    valid = moving & (lengths > length_eps)

    # “单位里程尖峰”
    jerk_pm = np.where(valid, jerk_p / (lengths + eps), np.nan)
    acc_pm  = np.where(valid, acc_p  / (lengths + eps), np.nan)
    yaw_pm  = np.where(valid, yaw_p  / (lengths + eps), np.nan)

    # —— 与 curvature 一致的归一化：S = 1/(1 + value) —— 可用 *scale 调整量纲
    S_j = 1.0 / (1.0 + (jerk_pm / max(j_scale, eps)))
    S_a = 1.0 / (1.0 + (acc_pm  / max(a_scale, eps)))
    S_y = 1.0 / (1.0 + (yaw_pm  / max(y_scale, eps)))

    # 几何平均（忽略 NaN 维），与前面风格一致
    comp = np.vstack([S_j, S_a, S_y])                       # (3,N)
    S_comf = np.exp(np.nanmean(np.log(comp), axis=0))       # (N,)

    # 还原批形状
    S_comf = S_comf.reshape(bs)
    if return_components:
        comps = np.stack([S_j, S_a, S_y], axis=-1).reshape(*bs, 3)  # (…,3)
        if reduce == "none":
            return S_comf, comps
        return float(np.nanmean(S_comf)), np.nanmean(comps, axis=tuple(range(comps.ndim-1)))
    else:
        if reduce == "none":
            return S_comf
        return float(np.nanmean(S_comf))


def curvature_rms(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,
    axis: int = -2,
    eps: float = 1e-9,
    v_static: float = 0.1,
    pct: float = None,          # NEW: 时间维保留的百分位
    reduce: str = "none",
) -> np.ndarray | float:
    """
    S_curv = 1 / (1 + κ_rms) ∈ (0,1]

    - 先算 κ(t)，再把 >pct 分位的尖峰丢掉 → 再 RMS
    - max speed < v_static → NaN
    """
    xy, bs = _prep_xy(np.asarray(traj_xy, float), axis)          # (N,T,2)
    N, T, _ = xy.shape
    if T < 3:
        raise ValueError("Need ≥3 frames to compute curvature")

    # ── 速度 & 静止判断 ─────────────────────────────────────────────
    v = np.diff(xy, axis=1) / dt                                 # (N,T-1,2)
    speed = np.linalg.norm(v, axis=-1)                           # (N,T-1)
    moving = speed.max(axis=1) >= v_static

    scores = np.full(N, np.nan, dtype=float)                     # 结果占位

    if moving.any():
        idx   = np.where(moving)[0]
        v_mv  = v[idx]                                           # (M,T-1,2)
        a_mv  = np.diff(v_mv, axis=1) / dt                       # (M,T-2,2)

        x_dot, y_dot = v_mv[..., 1:, 0], v_mv[..., 1:, 1]        # (M,T-2)
        x_dd , y_dd  = a_mv[..., :, 0],  a_mv[..., :, 1]         # (M,T-2)

        num   = np.abs(x_dot * y_dd - y_dot * x_dd)
        den   = (x_dot**2 + y_dot**2 + eps) ** 1.5
        kappa = num / den                                        # (M,T-2)

        # ── NEW: 95-th percentile过滤 ────────────────────────────
        if pct is not None:
            thr   = np.percentile(kappa, pct, axis=-1, keepdims=True)   # (M,1)
            mask  = kappa <= thr + eps
            kappa = np.where(mask, kappa, np.nan)                       # 尖峰→NaN

        rms = np.sqrt(np.nanmean(kappa**2, axis=-1))            # (M,)
        scores[idx] = 1.0 / (1.0 + rms)

    out = scores.reshape(bs)
    return out if reduce == "none" else float(np.nanmean(out))


def speed_score(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,          # 10 Hz → 0.1 s
    axis: int = -2,
    v_ref: float = 6.0,       # m/s ≈22 km/h
    k: float = 2.5,           # v_max / v_ref
    v_static: float = 0.1,    # 静止阈值 & v_min
    use_percentile=None,      # None→mean；或 90, 95…
    reduce: str = "none",
):
    """
    S_speed ∈ (0,1)，对数–线性映射，鼓励“起步快”：
        v_min = v_static
        v_max = k * v_ref
        S = ln(1 + v_stat / v_min) / ln(1 + v_max / v_min)
    """
    xy, bs = _prep_xy(np.asarray(traj_xy, float), axis)        # (N,T,2)
    v = np.linalg.norm(np.diff(xy, axis=1) / dt, axis=-1)      # (N,T-1)

    # -------- 轨迹单值统计 --------
    v_stat = (v.mean(axis=1) if use_percentile is None
              else np.percentile(v, use_percentile, axis=1))

    # -------- 仅对曾经移动过的轨迹评分 --------
    moving = v.max(axis=1) >= v_static
    scores = np.full_like(v_stat, np.nan, dtype=float)

    # -------- log-linear 映射 --------
    v_max  = k * v_ref

    denom  = np.log1p(v_max)

    num    = np.log1p(v_stat[moving])
    scores[moving] = np.clip(num / denom, 0.0, 1.0)
    
    # 在函数末尾，计算完 scores 后：
    scores[~moving] = 0.0   # 原先是 NaN

    # -------- 输出格式保持一致 --------
    out = scores.reshape(bs)
    return out if reduce == "none" else float(np.nanmean(out))


def get_traj_quality(preds):
    comfort_norm_ = comfort_score_norm(preds, reduce='none')
    crms = curvature_rms(preds, reduce='none')
    ss = speed_score(preds, reduce='none')

    arr = np.stack([comfort_norm_, crms, ss], axis=-1).astype(float) 
    quality = np.nanmean(arr, axis=-1)
    quality = np.nanmean(quality)
    return float(quality)