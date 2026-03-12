import argparse
import cv2
import glob
import matplotlib
matplotlib.use('Agg')           # 置于 import pyplot 前
import matplotlib.pyplot as plt
import numpy as np
import os

import numpy as np
from PIL import Image

import sys
import math
import random
import json

import torch
import pickle
from numpy.typing import ArrayLike

from trajs.traj_distribution import get_ftd
from trajs.traj_quality import get_traj_quality
from trajs.traj_consistency import get_traj_consistency
from trajs.traj_alignment import get_ade, get_dtw

def get_video(video_path):
    video_frames = VideoReader(video_path)
    video_frames = [(torch.from_numpy(v.asnumpy()).to(torch.float32)) / 255. for v in video_frames]
    video_frames = torch.stack(video_frames)
    return video_frames

def get_imgs(video_path):
    imgs = os.listdir(video_path)
    imgs = [os.path.join(video_path, f) for f in imgs]
    imgs = sorted(imgs)
    return imgs


def print_sheet_row(metrics, include_header=False):
    """打印一行 =SPLIT("..."," ,") 公式，粘到 Google Sheet 自动分列。"""
    q   = metrics.get('quality', {}) or {}
    obj = q.get('objective_quality', {}) or {}
    sm  = q.get('smoothness', []) or []

    mse   = sm[0] if len(sm) > 0 else None
    ssim  = sm[1] if len(sm) > 1 else None
    lpips = sm[2] if len(sm) > 2 else None
    cta   = obj.get('contrast_transfer_accuracy', obj.get('contrast_transfer accuracy'))

    cols = [
        ('FVD', metrics.get('distribution', {}).get('fvd')),
        ('mse', mse),
        ('ssim', ssim),
        ('lpips', lpips),
        ('flow', q.get('magnitude')),
        ('frame_dynamic_range_proxy', obj.get('frame_dynamic_range_proxy')),
        ('mtf50', obj.get('mtf50')),
        ('mtf10', obj.get('mtf10')),
        ('contrast_transfer_accuracy', cta),
        ('edge_rise_time', obj.get('edge_rise_time')),
        ('total_distortion', obj.get('total_distortion')),
        ('flare_attenuation', obj.get('flare_attenuation')),
        ('gradient_entropy', obj.get('gradient_entropy')),
        ('blur_extent', obj.get('blur_extent')),
        ('chroma_aberration', obj.get('chroma_aberration')),
        ('sequence_dynamic_range_proxy', obj.get('sequence_dynamic_range_proxy')),
        ('fmp_alias', obj.get('fmp_alias')),
        ('mmp_alias', obj.get('mmp_alias')),
        ('subjective_quality', q.get('subjective_quality')),
        ('scene_consistency', q.get('scene_consistency')),
    ]

    def fmt(v):
        if isinstance(v, (int, float)):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    headers = ",".join(k for k, _ in cols)
    values  = ",".join(fmt(v) for _, v in cols)

    if include_header:
        print(f'=SPLIT("{headers}", ",")')
    print(f'=SPLIT("{values}", ",")')


def print_by_metric(all_results: dict,
                    floatfmt: str = ".4f",
                    models_order: list[str] | None = None):
    """
    按 metric 聚合打印（严格按 models_order 排列列，避免错列）
    结构:
      all_results = {
        "cosmos": {"objective_quality": {...}},
        "wan":    {"objective_quality": {...}},
        ...
      }
    """
    # 1) 确定列顺序（默认保留字典插入顺序；也可显式传入）
    if models_order is None:
        models = list(all_results.keys())              # 保留原始插入顺序
    else:
        models = [m for m in models_order if m in all_results]  # 严格按给定顺序
        # 把未在 models_order 中、但 all_results 里存在的模型追加在末尾（可按需保留/删除）
        models += [m for m in all_results.keys() if m not in models]

    # 2) 汇总所有类别
    categories = sorted({cat for m in models for cat in all_results[m].keys()})

    for cat in categories:
        # 3) 行 = metric；列 = 模型
        row_map = {}
        for m in models:
            for met, v in all_results[m].get(cat, {}).items():
                row_map.setdefault(met, {})[m] = v
        if not row_map:
            continue

        metrics = sorted(row_map.keys())

        # 4) 预格式化为字符串，确保列对齐 & 不错列
        table = []  # 每行：[metric_name, val_cosmos, val_wan, ...]
        for met in metrics:
            row = [met]
            for m in models:
                v = row_map[met].get(m, None)
                if isinstance(v, (int, float)):
                    s = format(v, floatfmt)
                elif v is None:
                    s = "-"
                else:
                    s = str(v)
                row.append(s)
            table.append(row)

        headers = ["metric"] + models
        # 5) 按列计算宽度（包含表头+数据），然后打印
        cols = list(zip(*([headers] + table)))
        widths = [max(len(x) for x in col) for col in cols]

        print(f"\n[{cat}]")
        print(" ".join(h.ljust(widths[i]) if i == 0 else h.rjust(widths[i])
                       for i, h in enumerate(headers)))
        print("-" * (sum(widths) + len(widths) - 1))
        for r in table:
            print(" ".join(r[i].ljust(widths[i]) if i == 0 else r[i].rjust(widths[i])
                           for i in range(len(headers))))

def gt_2_ego(gt_xy, heading=None, k_ahead=1, min_step=1):
    # theta = torch.tensor(-yaw, dtype=torch.float32)
    gt      = torch.from_numpy(gt_xy)            # shape [T, 2]

    # ---- 平移到原点 --------------------------------------------------------
    origin  = gt[0]
    rel_gt  = gt - origin

    # 用多帧位移稳健估计速度朝向
    k = min(k_ahead, len(gt)-1)
    v = gt[k] - gt[0]
    if torch.linalg.norm(v) < min_step:
        # 找到第一个位移够大的帧
        for j in range(1, len(gt)):
            v = gt[j] - gt[0]
            if torch.linalg.norm(v) >= min_step:
                break

    # ---- 旋转：将 heading 对齐到 +Z ---------------------------------------
    if heading is None:
        # heading = rel_gt[1]                 # (Δx, Δy)
        heading = v
        theta   = torch.atan2(heading[1], heading[0])  # 车头相对 +x 的角度
    else:
        theta = heading
    R = torch.tensor([[ torch.cos(theta), -torch.sin(theta)],   # 逆时针旋转
                    [ torch.sin(theta),  torch.cos(theta)]])  # shape [2,2]

    gt_local = torch.matmul(rel_gt, R)
    gt_local_xy = gt_local.clone()
    gt_local[:, [0,1]] = gt_local[:, [1, 0]]
    gt_local[:, 0] = -gt_local[:, 0]
    gt = gt_local.numpy()

    # xy for mtr enc and yx for vis
    return gt_local_xy, gt, theta


from scipy.signal import savgol_filter

def smooth_traj_sg(xy, dt=0.1, win_sec=0.4, poly=3):
    """
    xy : (T, 2) 轨迹，单位米
    dt : 采样间隔 (s)
    win_sec : 滤波窗口秒数
    poly : 多项式阶次
    返回:
        xy_s  : 平滑后位置 (T,2)
    """
    xy = np.asarray(xy, float)
    T  = xy.shape[0]

    # SG 要求：window_length 为奇数，且  window_length <= T，且 window_length >= poly + 2
    k = int(round(win_sec / dt))
    if k % 2 == 0:
        k += 1                              # 强制奇数

    # —— 特定处理 1：窗口不能比序列长 ——
    if k > T:
        k = T if T % 2 == 1 else T - 1      # 截到 <=T 的最近奇数

    # —— 特定处理 2：T == 4 时官方约束变为 k=3, poly<=1 —
    if T == 4:
        k = 3
        poly = 1
    # 若一般情况下 poly 仍 ≥ k-1，也做自动降阶
    elif poly >= k - 1:
        poly = max(1, k - 2)

    # 轨迹太短无法平滑，直接返回原始
    if k < 3:
        return xy

    xy_s = savgol_filter(xy, window_length=k, polyorder=poly, axis=0, mode="interp")
    return xy_s

def ego_y_2_x(ego_xz):
    # how about angle?
    gt_local = torch.from_numpy(np.asarray(ego_xz,   float))
    gt_local[:, [0,1]] = gt_local[:, [1, 0]]
    gt_local[:, 1] = -gt_local[:, 1]
    gt = gt_local.numpy()
    return gt

def umeyama_2d(X: np.ndarray, Y: np.ndarray, with_scale=True):
    """
    X, Y : (N,2)   N ≥ 2，已按点序对齐（可先用最近帧或双向 DTW 对齐）
    返回   : s, R(2×2), t(2,)   使 Y ≈ s·R·X + t
    """
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc   = X - muX, Y - muY

    H = Xc.T @ Yc / len(X)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:          # 解决反射
        Vt[1] *= -1
        R = Vt.T @ U.T

    if with_scale:
        varX = (Xc**2).sum() / len(X)
        s = (Yc * (R @ Xc.T).T).sum() / (len(X) * varX)
    else:
        s = 1.0

    t = muY - s * R @ muX
    return s, R, t

def slam_align_to_gt_fix_origin(
    pred_xyz:   ArrayLike,      # (N,3)  SLAM: (x_right , y , z_forward)
    gt_local:   ArrayLike,      # (N,2)  来自 gt_2_ego → 首帧 = (0,0)
    with_scale: bool = True,
) -> np.ndarray:
    """
    只估计比例 s 和旋转 R，使得首帧保持 (0,0)。
    返回 (N,2) 列 0=X_right, 列 1=Z_forward
    """
    P = np.asarray(pred_xyz, float)   # 取地面分量 (x,z)
    Q = np.asarray(gt_local, float)

    # ---- 1. 去掉首帧平移，保证两条轨迹第 0 帧都在原点 ----
    P_rel = P - P[0]
    Q_rel = Q                                    # 已经在原点

    # ---- 2. SVD 求最佳旋转 R ------------------------------
    H = P_rel.T @ Q_rel / len(P_rel)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T

    # ---- 3. 比例 s（可选） -------------------------------
    if with_scale:
        varP = (P_rel**2).sum() / len(P_rel)
        s = (Q_rel * (R @ P_rel.T).T).sum() / (len(P_rel) * varP)
    else:
        s = 1.0

    # ---- 4. 应用 s·R，并把首帧重新放回原点 ---------------
    aligned = (s * (R @ P_rel.T)).T             # 仍以首帧为 (0,0)
    return aligned, s, R

def apply_sr(trajectory_xz, s, R):
    """
    trajectory_xz : (N,2) or (N,3)  与“原始 pred”同系 (x_right, z_forward)
    s, R          : 来自 slam_align_to_gt_fix_origin
    返回          : 已对齐到 GT 局部系 (X_right, Z_forward)
    """
    arr = np.asarray(trajectory_xz, float)

    aligned = (s * (R @ arr.T)).T
    return aligned

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unidepth')
    
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--model_name', type=str, default='gt')
    parser.add_argument('--exp_id', type=str, default='free')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--metric', type=str, default='fvd')
    parser.add_argument('--track', type=str, default='ego_condition')
    args = parser.parse_args()

    all_metrics = {}

    model_name = args.model_name
    exp_id = args.exp_id

    print(f'eval: {model_name}')

    model_name = model_name
    exp_id = exp_id if model_name != 'gt' else ''
    runs = os.listdir(args.root_path)
    runs = [run for run in runs if os.path.isdir(os.path.join(args.root_path, run))]
    debug = args.debug

    print(f'all {len(runs)} runs')

    gt_json = args.gt_path
    with open(gt_json, 'r') as f:
        gt_json = json.load(f)

    preds = []
    # gts = []

    pred_imgs = []
    gt_imgs = []

    gt_traj_path_base = 'data/ego_condition/ego_motion'

    if args.metric == 'ftd' or args.metric == 'all':
        gt_trajs_ftd = []
        for s_name in gt_json:
            gt_path = os.path.join(gt_traj_path_base, s_name+'.npy')
            gt = np.load(gt_path, allow_pickle=True)
            gt_local_xy, gt_local_yx, theta = gt_2_ego(gt[:101, :2])
            if True:
                gt_local_xy = smooth_traj_sg(gt_local_xy, dt=0.1, win_sec=0.4, poly=3)
            gt_trajs_ftd.append(gt_local_xy)
    # if args.metric == 'a_consist' or args.metric == 'a_missing' or args.metric == 'all':
    #     agents_bbox = []
    #     agents_label = []
    #     valid_agents_runs = []
    #     img_dirs = []
    if args.track == 'ego_condition':
        gt_trajs_align = []

    preds = []
    for run in runs:
        # if 'stationary' in run:
        #     print(f'ignore stationary {run}')
        #     continue
        s_name = run

        # root_path stored infer video
        # outdir stored trajs
        log_base = os.path.join(args.outdir, s_name, model_name, exp_id, 'unidepth')

        if args.track == 'ego_condition':
            gt_path = os.path.join(gt_traj_path_base, s_name+'.npy')
            gt = np.load(gt_path, allow_pickle=True)
            gt_local_xy, gt_local_yx, theta = gt_2_ego(gt[:101, :2])


        # video_path = os.path.join(log_base, 'video.mp4')
        # video_frames = get_video(video_path)
        # preds.append(video_frames)

        # video_path = os.path.join(log_base, 'images')
        # video_frames = get_imgs(video_path)
        # pred_imgs.append(video_frames[1:])

        # load traj and deal
        with open(log_base+'-estimate_ego_traj.pkl', 'rb') as f:
            data = pickle.load(f)
            pred = data['locs'].astype(np.float32)
        
        pred_xy = ego_y_2_x(pred)

        if args.track == 'ego_condition':
            pred_xy, s, r = slam_align_to_gt_fix_origin(pred_xy, gt_local_xy, with_scale=False)
            pred_xy = smooth_traj_sg(pred_xy, dt=0.1, win_sec=0.4, poly=3)
            gt_local_xy = smooth_traj_sg(gt_local_xy, dt=0.1, win_sec=0.4, poly=3)
            gt_trajs_align.append(gt_local_xy)
        else:
            pred_xy = smooth_traj_sg(pred_xy, dt=0.1, win_sec=0.4, poly=3)
        
        preds.append(pred_xy)

    preds = np.array(preds)

    ftd = -1
    if args.metric == 'ftd' or args.metric == 'all':
        gt_trajs_ftd = np.array(gt_trajs_ftd)
        ftd = get_ftd(preds, gt_trajs_ftd, stride=10)

    traj_quality = -1
    if args.metric == 'traj_q' or args.metric == 'all':
        traj_quality = get_traj_quality(preds)
    
    traj_consistency = -1
    if args.metric == 'traj_consist' or args.metric == 'all':
        traj_consistency = get_traj_consistency(preds)

    traj_ade = -1
    traj_dtw = -1
    if args.metric == 'traj_align' or args.metric == 'all':
        traj_ade = get_ade(preds, gt_trajs_align)
        traj_dtw = get_dtw(preds, gt_trajs_align)


    # 2. log to a dict: with hieral keys
    metrics = {
        'ftd': ftd,
        # 'smoothness': video_smoothness,
        # 'magnitude': video_magnitude,
        'traj_quality': traj_quality,
        'traj_consistency': traj_consistency,
        'traj_ade': traj_ade,
        'traj_dtw': traj_dtw
    }
    all_metrics[model_name] = metrics
    # if debug and idx == len(args.model_name) - 1:
    #     all_metrics['gt'] = {
    #         # 'fvd': fvd,
    #         # 'smoothness': gt_video_smoothness,
    #         # 'magnitude': video_magnitude,
    #         # 'objective_quality': gt_objective_quality,
    #         # 'subjective_quality': subjective_quality,
    #         'scene_consistency': gt_scene_consistency
    #     }

    for sub_key, sub_val in all_metrics.items():
        if isinstance(sub_val, (float, int)):        # 标量可用 .4f
            print(f"  {sub_key}: {sub_val:.4f}")
        else:                                        # 列表 / 其它类型直接打印
            print(f"  {sub_key}: {sub_val}")

    # print_by_metric(all_metrics)