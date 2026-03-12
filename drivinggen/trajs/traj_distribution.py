

# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import sys
sys.path.append('third_parties/MTR')
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.models import model as model_utils
from mtr.utils import common_utils


def parse_config():
    cfg_file = 'third_parties/MTR/tools/cfgs/waymo/mtr+100_percent_data.yaml'
    cktp = 'ckpt/mtr-epoch=28-step=176552.ckpt'

    cfg_from_yaml_file(cfg_file, cfg)

    np.random.seed(1024)

    return cfg


def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
                angle=-center_heading
            ).view(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs


def generate_centered_trajs_for_agents(center_objects, obj_trajs_past, obj_types, center_indices, sdc_index, timestamps, obj_trajs_future):
    """[summary]

    Args:
        center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        obj_trajs_past (num_objects, num_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        obj_types (num_objects):
        center_indices (num_center_objects): the index of center objects in obj_trajs_past
        centered_valid_time_indices (num_center_objects), the last valid time index of center objects
        timestamps ([type]): [description]
        obj_trajs_future (num_objects, num_future_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
    Returns:
        ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
        ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
        ret_obj_trajs_future (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
        ret_obj_valid_mask_future (num_center_objects, num_objects, num_timestamps_future):
    """
    # n_c, 10
    # n, t, 10
    '''
    最后一个点的10  轨迹11 type选vehicle    center选1        sdc选空    timestamps间隔0.1s  future不用管
    '''
    assert obj_trajs_past.shape[-1] == 10
    assert center_objects.shape[-1] == 10
    num_center_objects = center_objects.shape[0]
    num_objects, num_timestamps, box_dim = obj_trajs_past.shape
    # transform to cpu torch tensor
    center_objects = torch.from_numpy(center_objects).float()
    obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
    timestamps = torch.from_numpy(timestamps)

    # transform coordinates to the centered objects
    obj_trajs = transform_trajs_to_center_coords(
        obj_trajs=obj_trajs_past,
        center_xyz=center_objects[:, 0:3],
        center_heading=center_objects[:, 6],
        heading_index=6, rot_vel_index=[7, 8]
    )

    ## generate the attributes for each object
    object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
    object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
    object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # TODO: CHECK THIS TYPO
    object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
    object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 3] = 1
    # object_onehot_mask[:, sdc_index, :, 4] = 1


    object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
    object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
    object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

    object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
    object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
    object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

    vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
    vel_pre = torch.roll(vel, shifts=1, dims=2)
    acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
    acce[:, :, 0, :] = acce[:, :, 1, :]

    ret_obj_trajs = torch.cat((
        obj_trajs[:, :, :, 0:6], 
        object_onehot_mask,
        object_time_embedding, 
        object_heading_embedding,
        obj_trajs[:, :, :, 7:9], 
        acce,
    ), dim=-1)

    ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps)  # TODO: CHECK THIS, 20220322
    ret_obj_trajs[ret_obj_valid_mask == 0] = 0

    # ##  generate label for future trajectories
    # obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
    # obj_trajs_future = transform_trajs_to_center_coords(
    #     obj_trajs=obj_trajs_future,
    #     center_xyz=center_objects[:, 0:3],
    #     center_heading=center_objects[:, 6],
    #     heading_index=6, rot_vel_index=[7, 8]
    # )
    # ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
    # ret_obj_valid_mask_future = obj_trajs_future[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # TODO: CHECK THIS, 20220322
    # ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0

    # return ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(), ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy()
    return ret_obj_trajs.cuda(), (ret_obj_valid_mask > 0).cuda()

def compute_heading_xy(xy: np.ndarray) -> np.ndarray:
    """
    xy: (N, 2) 轨迹坐标
    return: (N,) 每帧 heading（弧度）
    """
    x, y = xy[:, 0], xy[:, 1]
    N = len(x)

    # 1. 中心差分得到中间段 heading（长度 N-2）
    dx_mid = x[2:] - x[:-2]
    dy_mid = y[2:] - y[:-2]
    heading_mid = np.arctan2(dy_mid, dx_mid)

    # 2. 首尾帧用前/后向差分
    heading_first = np.arctan2(y[1] - y[0], x[1] - x[0])   # 第 1 帧
    heading_last  = np.arctan2(y[-1] - y[-2], x[-1] - x[-2])  # 最后一帧

    # 3. 拼接
    heading = np.empty(N)
    heading[1:-1] = heading_mid
    heading[-1] = heading_last
    heading[0] = heading_first

    return heading


def compute_speed_xy(xy: np.ndarray, DT=0.1):
    """
    xy: (N, 2) 轨迹坐标 [x, y]
    返回:
        vx, vy: (N,) 逐帧速度分量
        v:      (N,) 逐帧速度标量 (magnitude)
    """
    x, y = xy[:, 0], xy[:, 1]
    N = len(x)

    # 1. 中心差分 —— 中间帧
    vx_mid = (x[2:] - x[:-2]) / (2 * DT)
    vy_mid = (y[2:] - y[:-2]) / (2 * DT)

    # 2. 前/后向差分 —— 首尾帧
    vx_first = (x[1] - x[0]) / DT
    vy_first = (y[1] - y[0]) / DT

    vx_last  = (x[-1] - x[-2]) / DT
    vy_last  = (y[-1] - y[-2]) / DT

    # 3. 拼接并做首帧复用
    vx = np.empty(N)
    vy = np.empty(N)

    vx[1:-1], vy[1:-1] = vx_mid, vy_mid
    vx[-1], vy[-1]     = vx_last, vy_last
    vx[0],  vy[0]      = vx_first, vy_first

    # 4. 标量速度
    v = np.hypot(vx, vy)                     # √(vx² + vy²)

    return vx, vy, v


def deal_pred_input(pred_traj_data):
    x = pred_traj_data[:, 0]
    y = pred_traj_data[:, 1]
    z = np.zeros_like(x)
    length = np.zeros_like(x) + 4.5
    width = np.zeros_like(x) + 2.0
    height = np.zeros_like(x) + 1.8
    heading = compute_heading_xy(pred_traj_data)
    # print(f'pred heading: {heading}')
    vx, vy, _ = compute_speed_xy(pred_traj_data)
    valid = np.ones_like(x)
    obj_trajs_past = np.stack((x, y, z, length, width, height, heading, vx, vy, valid), axis=-1)
    center_objects = obj_trajs_past[-1]
    obj_types = np.array('TYPE_VEHICLE').reshape(1)  # 假设只有一个中心对象，类型为车辆
    center_indices = np.array([0]).astype(np.int32)
    sdc_index = None
    timestamps = np.arange(len(x)).astype(np.float32) * 0.1
    obj_trajs_future = None

    return center_objects[None], obj_trajs_past[None], obj_types, center_indices, sdc_index, timestamps, obj_trajs_future


def gt_2_ego(gt_xy, yaw):
    theta = torch.tensor(-yaw, dtype=torch.float32)
    gt      = torch.from_numpy(gt_xy)            # shape [T, 2]

    # ---- 平移到原点 --------------------------------------------------------
    origin  = gt[0]
    rel_gt  = gt - origin

    # ---- 旋转：将 heading 对齐到 +Z ---------------------------------------
    # heading = rel_gt[1]                 # (Δx, Δy)
    # theta   = torch.atan2(heading[1], heading[0])  # 车头相对 +x 的角度
    R = torch.tensor([[ torch.cos(theta), torch.sin(theta)],   # 逆时针旋转
                    [ -torch.sin(theta),  torch.cos(theta)]])  # shape [2,2]

    gt_local = torch.matmul(rel_gt, R)             # shape [T, 2]，列 0=Z，列 1=X
    # 如果你想画 Z 在纵轴、X 在横轴，可直接：
    # Z, X = gt_local[:,0].numpy(), gt_local[:,1].numpy()
    gt_local[:, [0,1]] = gt_local[:, [1, 0]]
    gt_local[:, 0] = -gt_local[:, 0]
    gt = gt_local.numpy()   # X in x, Z in y

    return gt


def deal_gt_input(gt_traj_data):
    x = gt_traj_data[:, 0]
    y = gt_traj_data[:, 1]
    z = np.zeros_like(x)
    length = np.zeros_like(x) + 4.5
    width = np.zeros_like(x) + 2.0
    height = np.zeros_like(x) + 1.8
    heading = compute_heading_xy(gt_traj_data)
    vx, vy, _ = compute_speed_xy(gt_traj_data)
    # print(f'gt heading: {heading}')
    valid = np.ones_like(x)
    obj_trajs_past = np.stack((x, y, z, length, width, height, heading, vx, vy, valid), axis=-1)
    center_objects = obj_trajs_past[-1]
    obj_types = np.array('TYPE_VEHICLE').reshape(1)  # 假设只有一个中心对象，类型为车辆
    center_indices = np.array([0]).astype(np.int32)
    sdc_index = None
    timestamps = np.arange(len(x)).astype(np.float32) * 0.1
    obj_trajs_future = None

    return center_objects[None], obj_trajs_past[None], obj_types, center_indices, sdc_index, timestamps, obj_trajs_future
    

@torch.no_grad()
def infer(preds, gts, stride=10):
    cfg = parse_config()

    import random, numpy as np, torch
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    model = model_utils.MotionTransformer(config=cfg.MODEL)
    model.load_state_dict(torch.load('ckpt/mtr-epoch=28-step=176552.ckpt', weights_only=False)['state_dict'])
    mlp = model.context_encoder.agent_polyline_encoder.cuda()
    mlp.eval()

    Fp = []
    Fg = []
    
    for pred_traj_data in preds:
    
        pred_this_feature = []

        # for idx in range(10, 100, 10):
        for idx in range(0, pred_traj_data.shape[0] - 11 + 1, stride):
            # print(f'Processing index: {idx}')
            this_pred_traj_data = pred_traj_data[idx:idx + 11]
            this_pred_traj_data = deal_pred_input(this_pred_traj_data)

            obj_trajs, obj_trajs_mask = generate_centered_trajs_for_agents(*this_pred_traj_data)
            # apply polyline encoder
            obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
            obj_polylines_feature = mlp(obj_trajs_in, obj_trajs_mask)[0,0]
            has_nan = torch.isnan(obj_polylines_feature).any()
            if has_nan:
                import pdb
                pdb.set_trace()
                continue
            pred_this_feature.append(obj_polylines_feature)

        # direct concat
        # pred_this_feature = torch.stack(pred_this_feature, dim=0).reshape(-1)
        # gt_this_feature = torch.stack(gt_this_feature, dim=0).reshape(-1)

        if len(pred_this_feature) < 1:
            continue

        # mean
        pred_this_feature = torch.stack(pred_this_feature, dim=0)
        # pred_this_feature_max = pred_this_feature.max(0).values
        # pred_this_feature_std = pred_this_feature.std(0)
        pred_this_feature_mean = pred_this_feature.mean(0)

        # pred_this_feature = torch.cat((pred_this_feature_max, pred_this_feature_std, pred_this_feature_mean), dim=-1)[512:]
        pred_this_feature = pred_this_feature_mean

        # cat to sample
        # pred_this_feature = torch.stack(pred_this_feature, dim=0)
        # gt_this_feature = torch.stack(gt_this_feature, dim=0)

        Fp.append(pred_this_feature.cpu().numpy())

    for gt_traj_data in gts:
    
        gt_this_feature = []

        # for idx in range(10, 100, 10):
        for idx in range(0, gt_traj_data.shape[0] - 11 + 1, stride):
            # print(f'Processing index: {idx}')
            this_gt_traj_data = gt_traj_data[idx:idx + 11]
            this_gt_traj_data = deal_gt_input(this_gt_traj_data)

            obj_trajs, obj_trajs_mask = generate_centered_trajs_for_agents(*this_gt_traj_data)
            # apply polyline encoder
            obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
            obj_polylines_feature = mlp(obj_trajs_in, obj_trajs_mask)[0,0]
            gt_this_feature.append(obj_polylines_feature)

        # direct concat
        # pred_this_feature = torch.stack(pred_this_feature, dim=0).reshape(-1)
        # gt_this_feature = torch.stack(gt_this_feature, dim=0).reshape(-1)

        gt_this_feature = torch.stack(gt_this_feature, dim=0)
        # gt_this_feature_max = gt_this_feature.max(0).values
        # gt_this_feature_std = gt_this_feature.std(0)
        gt_this_feature_mean = gt_this_feature.mean(0)

        # gt_this_feature = torch.cat((gt_this_feature_max, gt_this_feature_std, gt_this_feature_mean), dim=-1)[512:]
        gt_this_feature = gt_this_feature_mean

        # cat to sample
        # pred_this_feature = torch.stack(pred_this_feature, dim=0)
        # gt_this_feature = torch.stack(gt_this_feature, dim=0)

        Fg.append(gt_this_feature.cpu().numpy())

    Fp = np.stack(Fp, axis=0).reshape(-1, Fp[0].shape[-1])
    Fg = np.stack(Fg, axis=0).reshape(-1, Fg[0].shape[-1])
    print(Fp.shape, Fg.shape)

    return Fp, Fg


import numpy as np
from scipy import linalg

def compute_fid_feats(
        X_real: np.ndarray,
        X_fake: np.ndarray,
        *,
        eps: float = 1e-6,        # 对角扰动大小
        unbiased: bool = True,    # 协方差是否用 N-1
        clip_negative: bool = True    # 结果微负时截 0
) -> float:
    """
    Robust CPU FID (Fréchet distance) implementation using scipy.linalg.sqrtm.

    Parameters
    ----------
    X_real, X_fake : array-like, shape (N, D) / (M, D)
        Feature vectors of real / generated samples.
    eps : float, default 1e-6
        Diagonal jitter added to covariance for numerical stability.
    unbiased : bool, default True
        If True, covariance is divided by N-1, matching官方 FID；
        False 等价于你原先的 bias=True。
    clip_negative : bool, default True
        将由数值噪声导致的微小负 FID 截断为 0。

    Returns
    -------
    fid : float
        Fréchet distance ≥ 0.
    """
    Xr = np.asarray(X_real, dtype=np.float64)
    Xg = np.asarray(X_fake, dtype=np.float64)

    # ---------- 均值 ----------
    mu_r = Xr.mean(axis=0)
    mu_g = Xg.mean(axis=0)
    diff = mu_r - mu_g

    # ---------- 协方差 ----------
    cov_opt = dict(rowvar=False, bias=not unbiased)
    sigma_r = np.cov(Xr, **cov_opt) + eps * np.eye(Xr.shape[1])
    sigma_g = np.cov(Xg, **cov_opt) + eps * np.eye(Xg.shape[1])

    # ---------- 协方差乘积的矩阵平方根 ----------
    cov_prod = sigma_r @ sigma_g
    covmean, _ = linalg.sqrtm(cov_prod, disp=False)

    # scipy 有时会返回复数; 仅保留实部
    if np.iscomplexobj(covmean):
        if not np.allclose(covmean.imag, 0, atol=1e-3):
            raise ValueError("sqrtm 返回显著虚部，可能协方差奇异")
        covmean = covmean.real

    # ---------- FID ----------
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean)

    # 数值噪声可能使 fid 落到 -1e-6 级别；按约定截断
    if clip_negative and fid < 0:
        fid = 0.0

    return float(fid)

def get_ftd(
    pred_traj,
    gt_traj,
    stride:int = 1,
    eps: float = 1e-6,
):
    # pred = np.asarray(pred_traj, dtype=float)
    # gt   = np.asarray(gt_traj,   dtype=float)

    Fp, Fg = infer(pred_traj, gt_traj, stride=stride)
    # Fp, Fg = infer_single_level(pred, gt, stride=stride)

    return compute_fid_feats(Fp, Fg)