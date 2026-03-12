"""traj_gt_metrics.py – Ground‑truth‑aware trajectory error metrics.

Each metric is a standalone NumPy function.  No external deps except an
optional SciPy import for DTW; a pure‑NumPy fallback is provided.

Conventions
-----------
* Prediction & GT arrays have the **same shape** ``(..., T, C)`` with
  :math:`C∈{2,3}`; leading batch dims are preserved in outputs.
* ``axis`` selects the time dimension (default ``-2``).

Implemented metrics
-------------------
1. ``ade``                 – Average Displacement Error
2. ``fde``                 – Final Displacement Error
3. ``success_rate``        – FDE < *threshold* ratio
4. ``hausdorff``           – Symmetric Hausdorff distance
5. ``ndtw``                – Normalised Dynamic Time‑Warping score
6. ``dynamic_consistency`` – Motion‑dynamic similarity (Wasserstein)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "ade",
    "fde",
    "success_rate",
    "hausdorff",
    "ndtw",
    "dynamic_consistency",
]

# ------------------------------------------------------------------
# 通用预处理
# ------------------------------------------------------------------
def _prep(a: np.ndarray, b: np.ndarray, axis: int):
    b = b[..., :2]
    if a.shape != b.shape:
        raise ValueError("pred and gt must share shape")

    a = np.moveaxis(a, axis, -2)          # (..., T, C)
    b = np.moveaxis(b, axis, -2)
    batch_shape = a.shape[:-2]            # e.g. (B,) or (B1,B2)
    T, C = a.shape[-2:]
    a = a.reshape(-1, T, C)               # (N,T,C)
    b = b.reshape(-1, T, C)
    return a, b, batch_shape

# ------------------------------------------------------------------
# 1. ADE / 2. FDE / 3. Success-rate
# ------------------------------------------------------------------
def ade(pred, gt, *, axis=-2, reduce="none"):
    p, g, bs = _prep(np.asarray(pred), np.asarray(gt), axis)
    err = np.linalg.norm(p - g, axis=-1).mean(-1)     # (N,)
    err = err.reshape(bs)
    return err if reduce == "none" else err.mean()

def fde(pred, gt, *, axis=-2, reduce="none"):
    p, g, bs = _prep(np.asarray(pred), np.asarray(gt), axis)
    err = np.linalg.norm(p[:, -1] - g[:, -1], axis=-1)
    err = err.reshape(bs)
    return err if reduce == "none" else err.mean()

def success_rate(pred, gt, *, threshold=3.0, axis=-2, reduce="none"):
    sr = (fde(pred, gt, axis=axis, reduce="none") < threshold)
    return sr if reduce == "none" else sr.mean()

# ------------------------------------------------------------------
# 4. Hausdorff  (仍保留逐样本 for 循环)
# ------------------------------------------------------------------
def hausdorff(pred, gt, *, axis=-2, reduce="none"):
    p, g, bs = _prep(np.asarray(pred), np.asarray(gt), axis)
    N = p.shape[0]
    out = np.empty(N)
    for i in range(N):
        D = np.linalg.norm(p[i, :, None] - g[i, None, :], axis=-1)
        out[i] = max(D.min(1).max(), D.min(0).max())
    out = out.reshape(bs)
    return out if reduce == "none" else out.mean()

# ------------------------------------------------------------------
# 5. nDTW  (依旧 O(T²) for-loop)
# ------------------------------------------------------------------
try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None

def _dtw(a: np.ndarray, b: np.ndarray):
    """Classic DTW cost with L2; inputs (T, C)."""
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf)
    D[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            dist = np.linalg.norm(a[i-1] - b[j-1])
            D[i, j] = dist + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return D[n, m]

def dtw(pred, gt, *, alpha=4.0, axis=-2, reduce="none"):
    p, g, bs = _prep(np.asarray(pred), np.asarray(gt), axis)
    N, T, _ = p.shape
    score = np.empty(N)
    for i in range(N):
        cost = _dtw(p[i], g[i])           # 复用原始 _dtw
        score[i] = cost
    score = score.reshape(bs)
    return score if reduce == "none" else score.mean()


import numpy as np         # 保证跟你前面的 import 一致

def sdtw(pred, gt, *, threshold=2.0, alpha=4.0,
         axis=-2, reduce="none"):
    """
    Success-weighted nDTW:
      • 先算 SR（二元 0/1）
      • 再算 nDTW (0~1)
      • 相乘得到 sDTW
    保留与 SR / nDTW 相同的 axis & reduce 语义
    """
    # ① SR，转 float 方便乘法
    sr = success_rate(pred, gt, threshold=threshold,
                      axis=axis, reduce="none").astype(float)
    # ② nDTW
    ndtw_score = ndtw(pred, gt, alpha=alpha,
                      axis=axis, reduce="none")
    # ③ 逐样本相乘
    # out = sr * ndtw_score
    out = ndtw_score

    return out if reduce == "none" else out.mean()


# ------------------------------------------------------------------
# 6. Dynamic-consistency (Wasserstein)  (for-loop)
# ------------------------------------------------------------------
def dynamic_consistency(pred, gt, *, dt=0.1, axis=-2, reduce="none"):
    """Return exp(-W_d_vel) · exp(-W_d_acc)  ∈(0,1].  Higher = dynamics closer.

    Uses 1‑D Wasserstein distance between speed & acceleration magnitude
    distributions.  No external hyper‑parameters.  If SciPy unavailable, falls
    back to empirical CDF integration.  Complexity O(N log N).
    """
    try:
        from scipy.stats import wasserstein_distance as w1
    except ImportError:
        def w1(x, y):
            x, y = np.sort(x), np.sort(y)
            allv = np.sort(np.concatenate([x, y]))
            cdfx = np.searchsorted(x, allv, side="right") / len(x)
            cdfy = np.searchsorted(y, allv, side="right") / len(y)
            return np.trapz(np.abs(cdfx - cdfy), allv)
    
    p, g, bs = _prep(np.asarray(pred), np.asarray(gt), axis)
    N, T, _ = p.shape
    out = np.empty(N)
    for i in range(N):
        v_p = np.linalg.norm(np.diff(p[i], axis=0) / dt, axis=-1)
        v_g = np.linalg.norm(np.diff(g[i], axis=0) / dt, axis=-1)
        a_p, a_g = np.diff(v_p), np.diff(v_g)
        dv = w1(v_p, v_g)
        da = w1(a_p, a_g)
        out[i] = np.exp(-dv) * np.exp(-da)
    out = out.reshape(bs)
    return out if reduce == "none" else out.mean()



def get_ade(preds, gts):
    ad_err = ade(preds, gts, reduce='none')
    ad_err = np.nanmean(ad_err)
    return float(ad_err)

def get_dtw(preds, gts):
    dtw_err = dtw(preds, gts, reduce='none')
    dtw_err = np.nanmean(dtw_err)
    return float(dtw_err)


# -----------------------------------------------------------------------------
# Quick demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    B = 8          # batch size
    T = 101
    t = np.linspace(0, 4 * np.pi, T)

    # (T,2) → (B,T,2)
    single_traj = np.stack([30 * np.cos(t), 30 * np.sin(t)], -1)      # (T,2)
    gt = np.repeat(single_traj[None, ...], B, axis=0)                 # (B,T,2)

    # 带噪预测，同样 (B,T,2)
    pred = (gt                                                  # 基础轨迹
            + 0.4 * np.random.randn(B, T, 2)                    # 噪声
            + 0.3 * np.roll(gt, 10, axis=1))                    # 时移

    # all use 0th information (pos)
    print("ADE:", ade(pred, gt, reduce='mean'))
    print("FDE:", fde(pred, gt, reduce='mean'))
    print("SR @2m:", success_rate(pred, gt, axis=-2, reduce='mean'))
    print("Hausdorff:", hausdorff(pred, gt, reduce='mean'))
    print("nDTW:", ndtw(pred, gt, reduce='mean'))
    # distribution based? ignore now
    # print("DynCons:", dynamic_consistency(pred, gt, reduce='mean'))

    '''
    ADE: 9.020381286932569
    FDE: 9.043563595143615
    SR @2m: 0.0
    Hausdorff: 5.470039531722291
    nDTW: 0.34880821820753677
    DynCons: 1.278691741265335e-06
    '''
