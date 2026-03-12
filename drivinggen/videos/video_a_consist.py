import torch
from typing import Tuple
import numpy as np
import math
import sys
from tqdm import tqdm
import os
import cv2
import torch.nn.functional as F
import torch.nn as nn
import pickle


import os, cv2, torch, numpy as np
from PIL import Image
from torchvision.transforms import (Compose, Resize, CenterCrop, ToTensor, Normalize)

# ───────────────────────── 安全裁剪 ─────────────────────────
def safe_crop_expand(frame: np.ndarray,
                     box,
                     min_size: int = 3):
    w, h = frame.size
    x1, y1, x2, y2 = map(int, box)

    # 1) 截到图像范围
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h)

    # 2) 若宽/高不足，外扩
    if x2 - x1 < min_size:
        need = min_size - (x2 - x1)
        left  = min(need // 2 + need % 2, x1)
        right = min(need // 2,           w - x2)
        x1 -= left
        x2 += need - left

    if y2 - y1 < min_size:
        need = min_size - (y2 - y1)
        up   = min(need // 2 + need % 2, y1)
        down = min(need // 2,           h - y2)
        y1 -= up
        y2 += need - up

    # 3) 最终 clip（确保宽、高 ≥1）
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, x1 + 1, w)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, y1 + 1, h)

    return frame.crop((x1, y1, x2, y2))  # RGB


######## use dinov3
dinov3 = None
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    # 旧版没有该函数时，返回 False 即可（与默认行为一致）
    torch.compiler.is_compiling = lambda: False
from modelscope import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image as load_image_hf

def init_dinov3_model():
    global dinov3
    # Load model, ray sampler, datasets
    if dinov3 is not None:
        return

    model_dir = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

    # 强制本地、允许自定义代码
    processor = AutoImageProcessor.from_pretrained(
        model_dir
    )
    model = AutoModel.from_pretrained(
        model_dir, device_map='auto'
    )
    dinov3 = model, processor

import pickle
import torch.nn as nn
# ─────────────────── 稳定性指标（以 boxes 为准） ───────────────────
def stability_metric(
    img_dir: str,
    boxes: list[tuple[int, tuple[int, int, int, int]]],
    label: str | None = None,
    embed_dir='',
    weights=(0.5, 0.5, 0),
):
    """
    img_dir : 帧目录，文件名 00000.png/.jpg
    boxes   : [(frame_id, (x1,y1,x2,y2)), ...] 需按时间顺序
    label   : 类别文本；None 则不计语义分量
    """
    if len(boxes) < 2:
        raise ValueError("需要至少 2 个框")

    global dinov3
    init_dinov3_model()
    model, processor = dinov3

    boxes = sorted(boxes, key=lambda x: x[0])
    w_R, w_A, w_S = weights
    if label is None:
        w_R, w_A = w_R / (w_R + w_A), w_A / (w_R + w_A)
        w_S = 0.0

    # ---- 图像嵌入 ----
    if not os.path.exists(embed_dir):

        embs_dino = []
        crops = []
        for fid, box in boxes:
            img_path = os.path.join(img_dir, f"{fid:05}.png")
            if not os.path.exists(img_path):
                img_path = img_path.replace('.png', '.jpg')
            # frame = cv2.imread(img_path)
            frame = Image.open(img_path)
            if frame is None:
                raise FileNotFoundError(img_path)
            crop = safe_crop_expand(frame, box, min_size=32)   # PIL RGB
            crops.append(crop)
            # embs_dino.append(dino_embed(crop))                 # 1024-d for R/A
            image_input = load_image_hf(crop)
            inputs = processor(images=image_input, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model(**inputs)
            scene_feat = outputs.pooler_output.squeeze(0)
            embs_dino.append(scene_feat)
        embs_dino = torch.stack(embs_dino)                     # (T,1024)
        print(f'store to: {embed_dir}')
        with open(embed_dir, 'wb') as f:
            # Pickle the dictionary and write it to the file
            pickle.dump(embs_dino.cpu().numpy(), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:

        with open(embed_dir, 'rb') as f:
            embs_dino = torch.from_numpy(pickle.load(f)).cuda()


    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # ---- 参考一致性 R（DINO）----
    sims_ref = cos(embs_dino, embs_dino[0:1]).clamp_min(0).cpu().numpy()
    R = float(sims_ref[1:].mean())

    # ---- 连续一致性 A（DINO）----
    sims_adj = cos(embs_dino[1:], embs_dino[:-1]).clamp_min(0).cpu().numpy()
    A = float(sims_adj.mean())

    # ---- 语义一致性 S（CLIP 图像 vs 文本）----
    if w_S > 0:
        # clip_embs = torch.stack([clip_image_embed(c) for c in crops])  # (T,512)
        # txt_vec = clip_text_embed(label).to(device)                     # (512,)
        # sims_txt = (clip_embs @ txt_vec).clamp_min(0).cpu().numpy()     # cosine ≥0
        # s0 = sims_txt[0]
        # RS = float((sims_txt[1:] / (s0 + 1e-6)).mean())
        pass
    else:
        RS = 0.0

    score = w_R * R + w_A * A + w_S * RS

    return {"R": R, "A": A, "S": RS, "score": score}


def max_iou_box(query_box: np.ndarray, boxes: np.ndarray, return_index: bool = False):
    """
    Find the box with the highest IoU w.r.t. `query_box`.
    
    Parameters
    ----------
    query_box : array-like, shape (4,)
        [x1, y1, x2, y2] for the reference box (x1 ≤ x2, y1 ≤ y2).
    boxes : array-like, shape (N, 4)
        Remaining boxes to compare against, each in [x1, y1, x2, y2] format.
    return_index : bool, default=False
        If True, also return the index of the best-IoU box in `boxes`.
        
    Returns
    -------
    best_box : np.ndarray, shape (4,)
        The box from `boxes` that maximises IoU with `query_box`.
    best_iou : float
        The corresponding IoU value.
    best_idx : int, optional
        Only if `return_index=True`.
    """
    query_box = np.asarray(query_box, dtype=float)
    boxes     = np.asarray(boxes,     dtype=float)
    if boxes.size == 0:
        raise ValueError("`boxes` is empty.")

    # Intersection coordinates
    ix1 = np.maximum(query_box[0], boxes[:, 0])
    iy1 = np.maximum(query_box[1], boxes[:, 1])
    ix2 = np.minimum(query_box[2], boxes[:, 2])
    iy2 = np.minimum(query_box[3], boxes[:, 3])

    # Intersection area (clip at 0)
    inter_w = np.clip(ix2 - ix1, 0, None)
    inter_h = np.clip(iy2 - iy1, 0, None)
    inter   = inter_w * inter_h

    # Areas
    query_area = (query_box[2] - query_box[0]) * (query_box[3] - query_box[1])
    boxes_area = (boxes[:, 2]  - boxes[:, 0]) * (boxes[:, 3]  - boxes[:, 1])

    # IoU
    iou = inter / (query_area + boxes_area - inter + 1e-8)

    best_idx  = int(iou.argmax())
    best_iou  = float(iou[best_idx])
    best_box  = boxes[best_idx]

    return (best_box, best_iou, best_idx) if return_index else (best_box, best_iou)


def get_agent_consistency(valid_agents_runs, agents_bbox, agents_label, names, img_dirs):
    print(f'len valid runs: {len(valid_agents_runs)}, len agents_bbox: {len(agents_bbox)}')
    scenes_stability = []
    for scene_id, scene_agents in tqdm(enumerate(agents_bbox)):
        labels_this_scene = agents_label[scene_id]
        img_dir = img_dirs[scene_id]
        name = names[scene_id]

        candidate_box = []
        for bbox in list(labels_this_scene.keys()):
            c_x1, c_y1, c_x2, c_y2 = bbox.split('-')
            c_x1 = int(c_x1)
            c_y1 = int(c_y1)
            c_x2 = int(c_x2)
            c_y2 = int(c_y2)
            candidate_box.append([c_x1, c_y1, c_x2, c_y2])
        candidate_box = np.array(candidate_box).astype(np.int32)

        scene_stability = []
        for agent_id, scene_agent in enumerate(scene_agents):
            if len(scene_agent) < 2:
                print(f'skip: {valid_agents_runs[scene_id]}, {scene_agent}')
                continue
            frame_0_bbox = scene_agent[0][1]

            agent_bbox = np.array(frame_0_bbox).astype(np.int32)
            try:
                match_box , _ = max_iou_box(agent_bbox, candidate_box)
            except:
                import pdb
                pdb.set_trace()
            match_box = match_box.astype(np.int32)
            key = f'{match_box[0]}-{match_box[1]}-{match_box[2]}-{match_box[3]}'
            label = labels_this_scene[key]
            # mask = (candidate_box == match_box).all(axis=1)   # shape (N,)
            # idx  = np.nonzero(mask)[0]                        # 匹配到的行索引

            # if idx.size:                                      # 至少找到一行
            #     candidate_box = np.delete(candidate_box, idx[0], axis=0)

            # video, bbox, label
            # if split == 'gt':
            #     img_dir = os.path.join(gt_paths[valid_agents_runs[scene_id]], 'CAM_F0')
            # else:
            #     img_dir = os.path.join(args.video_path, valid_agents_runs[scene_id], split, act_dir, 'images')

            unique = valid_agents_runs[scene_id]
            embed_dir = os.path.join(name, unique)

            os.makedirs(embed_dir, exist_ok=True)
            embed_dir = os.path.join(name, unique, f'{agent_id}.pkl')
            obj_stablity = stability_metric(img_dir, scene_agent, label, embed_dir)
            first_sim = obj_stablity['R']
            adj_sim = obj_stablity['A']
            text_rel_sim = obj_stablity['S']
            ss = obj_stablity['score']
            scene_stability.append([first_sim, adj_sim, text_rel_sim, ss])
        scene_stability = np.array(scene_stability)
        scene_stability = np.nanmean(scene_stability, axis=0)
        scenes_stability.append(scene_stability)
    scenes_stability = np.array(scenes_stability)
    scenes_stability = np.nanmean(scenes_stability, axis=0)
    return float((scene_stability[0] + scene_stability[1]) / 2)   

    metrics['object_stability'] = scenes_stability
    print(metrics)