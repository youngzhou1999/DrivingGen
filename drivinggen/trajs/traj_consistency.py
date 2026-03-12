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

def trajectory_consistency(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,
    axis: int = -2,
    eps: float = 1e-9,
    v_static: float = 0.1,
    reduce: str = "none",
) -> np.ndarray | float:
    """
    速度 + 加速度一致性评分
        S = exp(−σ_v / (μ_v + eps)) · exp(−σ_a / (μ_a + eps))
    若整条轨迹 max speed < v_static → NaN
    """
    xy, bs = _prep_xy(np.asarray(traj_xy, float), axis)          # (N,T,2)
    v = np.linalg.norm(np.diff(xy, axis=1) / dt, axis=-1)        # (N,T-1)

    scores = np.full(v.shape[0], np.nan, dtype=float)            # 先全部 NaN
    moving = v.max(axis=1) >= v_static                           # (N,)
    # print(f'consistency: Moving: {moving.sum()} / {len(moving)}')  # 打印非静止数量

    if moving.any():
        v_m = v[moving]                                          # (M,T-1)
        mu_v = v_m.mean(axis=1)
        sigma_v = v_m.std(axis=1)
        scores[moving] = np.exp(-sigma_v / (mu_v + eps))

        a_m = np.diff(v_m, axis=1) / dt                          # (M,T-2)
        mu_a = np.mean(np.abs(a_m), axis=1)
        sigma_a = np.std(a_m, axis=1)
        scores[moving] += np.exp(-sigma_a / (mu_a + eps))
        scores[moving] *= 0.5

    out = scores.reshape(bs)
    return out if reduce == "none" else float(np.nanmean(out))

def get_traj_consistency(preds):
    tc = trajectory_consistency(preds, reduce='none')
    tc = np.nanmean(tc)
    return float(tc)