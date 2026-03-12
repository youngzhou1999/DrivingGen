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

from decord import VideoReader
from videos.video_distribution import get_fvd
from videos.video_obj_q import get_objective_quality_v2
from videos.video_sub_q import get_subjective_quality
from videos.video_v_consist import get_scene_consistency_v3
from videos.video_a_consist import get_agent_consistency
from videos.video_a_missing import get_agent_missing
# from videos.video_quality import (
#     get_lpips, get_psnr, get_ssim, get_video_smoothness, get_video_magnitude, get_video_smoothness_v2,
#     get_objective_quality, get_subjective_quality, get_scene_consistency, get_scene_consistency_v2, get_scene_consistency_v3,
#     get_objective_quality_v2
# )

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unidepth')
    
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--model_name', type=str, default='gt')
    parser.add_argument('--exp_id', type=str, default='free')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--metric', type=str, default='fvd')
    args = parser.parse_args()

    all_metrics = {}

    model_name = args.model_name
    exp_id = args.exp_id

    print(f'eval: {model_name}')

    exp_id = exp_id if model_name != 'gt' else ''
    runs = os.listdir(args.root_path)
    runs = [run for run in runs if os.path.isdir(os.path.join(args.root_path, run))]
    debug = args.debug

    print(f'all {len(runs)} runs')

    gt_json = args.gt_path
    with open(gt_json, 'r') as f:
        gt_json = json.load(f)

    preds = []

    pred_imgs = []
    gt_imgs = []

    if args.metric == 'fvd' or args.metric == 'all':
        gt_video_fvd_base = "data/videos-fvd"
        video_fvd_base = args.root_path + f'+{model_name}_fvd'
        os.makedirs(video_fvd_base, exist_ok=True)
    if args.metric == 'a_consist' or args.metric == 'a_missing' or args.metric == 'all':
        agents_bbox = []
        agents_label = []
        valid_agents_runs = []
        img_dirs = []
    for run in runs:
        # if 'stationary' in run:
        #     print(f'ignore stationary {run}')
        #     continue
        s_name = run
        log_base = os.path.join(args.root_path, s_name, model_name, exp_id)

        # video_path = os.path.join(log_base, 'video.mp4')
        # video_frames = get_video(video_path)
        # preds.append(video_frames)

        video_path = os.path.join(log_base, 'images')
        video_frames = get_imgs(video_path)
        pred_imgs.append(video_frames[1:])
    
        # if not os.path.exists(video_fvd_base):
        if args.metric == 'fvd' or args.metric == 'all':
            name = log_base.split('/')[-3] + '+' + log_base.split('/')[-2] + '+' + log_base.split('/')[-1]
            fvd_path = os.path.join(video_fvd_base, name)
            if os.path.exists(fvd_path):
                os.system(f'rm -rf {fvd_path}')
            os.makedirs(fvd_path, exist_ok=True)
            for idx, img in enumerate(video_frames):
                # if idx == 0:
                #     continue
                # fvd_path_i = os.path.join(fvd_path, f'{idx:05d}.png')
                # os.system(f'cp {img} {fvd_path_i}')
                
                # 跳过第一个帧
                if idx == 0:
                    continue
                
                # 软连接的目标路径
                link_path = os.path.join(fvd_path, f'{idx:05d}{img[-4:]}')
                
                # 检查软连接是否已存在，防止重复创建时报错
                if not os.path.exists(link_path):
                    # 创建软连接，img是源文件，link_path是软连接
                    os.symlink(os.path.abspath(img), link_path)
        
        if args.metric == 'a_consist' or args.metric == 'a_missing' or args.metric == 'all':

            outdir = os.path.join(args.outdir, s_name, model_name, exp_id, 'unidepth')
            try:
                with open(outdir+'-estimate_agents_bbox.pkl', 'rb') as f:
                    agent_bbox = pickle.load(f)
                with open(outdir+'-estimate_agents_bbox_label.pkl', 'rb') as f:
                    agent_label = pickle.load(f)
            except:
                print(f"Skipping agents: {outdir}.")
                agent_bbox = None

            if agent_bbox is not None:
                ## newly updated: trim box
                trans_agent_bbox = []
                for box_id, boxes in enumerate(agent_bbox):
                    ids = [box[0] for box in boxes]
                    # keep the first segment
                    ids_valid = []
                    boxes_valid = [] 
                    for i, id in enumerate(ids):
                        ids_valid.append(id)
                        boxes_valid.append(boxes[i])
                        if i < len(ids) - 1:
                            if id + 10 < ids[i+1]:
                                print(f'trim {ids} at {id}')
                                break
                    boxes = boxes_valid
                    ids = ids_valid
                    # print(f'len agent {ids}: {len(traj)}')
                    # no filt
                    # if len(boxes) <= 10:
                    #     continue
                    trans_agent_bbox.append(boxes)

                # agents_traj.append(trans_agent_traj)
                # need to reconsider here: do we compute the first consective segment or all segments
                # agents_bbox.append(agent_bbox)
                agents_bbox.append(trans_agent_bbox)
                agents_label.append(agent_label)
                # gt_agents_traj.append(gt_agent_traj)
                # gt_agents_bbox.append(gt_agent_bbox)
                # vis_traj = [t[1] for t in trans_agent_traj]
                valid_agents_runs.append(run)
                img_dirs.append(os.path.join(log_base, 'images'))

    # preds = torch.stack(preds)   # n t c h w
    # gts = torch.stack(gts)

    # 1. get all metrics
    # fid = spectral_fid(preds, gts, level="speed", k=16)
    # m_fid = multi_spectral_fid(preds, gts, levels=("speed", "acc", "jerk"), k=64)
    fvd = -1
    # fvd = get_fvd(preds, gts)
    if args.metric == 'fvd' or args.metric == 'all':
        fvd = get_fvd(video_fvd_base, gt_video_fvd_base)

    objective_quality = -1
    gt_objective_quality = -1

    if args.metric == 'obj_q' or args.metric == 'all':
        objective_quality = get_objective_quality_v2(pred_imgs)
    # if debug:
        # gt_objective_quality = get_objective_quality_v2(gt_imgs)
    # noise = torch.rand_like(gts)
    # objective_quality = get_objective_quality(noise)

    # objective_quality = get_objective_quality(preds)

    subjective_quality = -1
    gt_subjective_quality = -1
    if args.metric == 'sub_q' or args.metric == 'all':
        subjective_quality = get_subjective_quality(pred_imgs)
    # if debug and idx == len(args.model_name) - 1:
    #     gt_subjective_quality = get_subjective_quality(gt_imgs)

    scene_consistency = -1
    gt_scene_consistency = -1
    if args.metric == 'v_consist' or args.metric == 'all':
        track = args.root_path.split('/')[-1]
        cache_dir = os.path.join('./cache/v_consist', track)
        os.makedirs(cache_dir, exist_ok=True)
        v_consist_cache_dir = [f'{cache_dir}/{name}' for name in runs]
        scene_consistency = get_scene_consistency_v3(pred_imgs, v_consist_cache_dir)
    # if debug and idx == len(args.model_name) - 1:
    #     gt_scene_consistency = get_scene_consistency_v3(gt_imgs)

    agent_consistency = -1
    if args.metric == 'a_consist' or args.metric == 'all':

        track = args.root_path.split('/')[-1]
        cache_dir = os.path.join('./cache/a_consist', track)
        os.makedirs(cache_dir, exist_ok=True)
        a_consist_cache_dir = [f'{cache_dir}/{name}' for name in valid_agents_runs]

        agent_consistency = get_agent_consistency(valid_agents_runs, agents_bbox, agents_label, a_consist_cache_dir, img_dirs)

    if args.metric == 'a_missing' or args.metric == 'all':

        track = args.root_path.split('/')[-1]
        cache_dir = os.path.join('./cache/a_missing', track)
        os.makedirs(cache_dir, exist_ok=True)
        a_missing_cache_dir = [f'{cache_dir}/{name}' for name in valid_agents_runs]

        agent_missing = get_agent_missing(valid_agents_runs, agents_bbox, agents_label, a_missing_cache_dir, img_dirs)


    # 2. log to a dict: with hieral keys
    metrics = {
        'fvd': fvd,
        # 'smoothness': video_smoothness,
        # 'magnitude': video_magnitude,
        'objective_quality': objective_quality,
        'subjective_quality': subjective_quality,
        'scene_consistency': scene_consistency,
        'agent_consistency': agent_consistency,
        'agent_missing': agent_missing
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