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


sys.path.append(os.path.abspath('third_parties/SEA-RAFT'))
sys.path.append(os.path.abspath('third_parties/SEA-RAFT/core'))
import argparse
from core.raft import RAFT
from core.utils.utils import load_ckpt
from config.parser import parse_args

flow_model = None
flow_args = None

def init_flow_model():
    global flow_model, flow_args
    if flow_model is not None:
        return
    
    args = {
        "cfg": "third_parties/SEA-RAFT/config/eval/kitti-M.json",
        "path": "ckpt/Tartan-C-T-TSKH-kitti432x960-M.pth",
    }
    args = argparse.Namespace(**args)
    args = parse_args(args)
    
    # load model
    model = RAFT(args)
    load_ckpt(model, args.path)
    model.to('cuda')
    model.eval()

    flow_model = model
    flow_args = args

def load_image(imfile):
    image = cv2.imread(imfile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = image[None].to('cuda')
    return image

def forward_flow(image1, image2):
    with torch.amp.autocast(device_type="cuda"):
        output = flow_model(image1, image2, iters=flow_args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def _compute_flow(image1, image2):

    img1 = F.interpolate(image1, scale_factor=2 ** flow_args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** flow_args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** flow_args.scale, mode='bilinear', align_corners=False) * (0.5 ** flow_args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** flow_args.scale, mode='area')
    
    flow = flow_down.cpu().numpy().squeeze().transpose(1, 2, 0)
    return flow

def compute_motion_series(images) -> Tuple[np.ndarray, float]:
    """返回每个相邻帧的光流幅值统计量 m_t 以及图像对角线长度（用于归一化）"""
    assert len(images) >= 2
    mags = []

    prev = images[0]
    for nxt in images[1:]:
        image1_f = load_image(prev)
        image2_f = load_image(nxt)
        
        flow = _compute_flow(image1_f, image2_f)
        flow_magnitude = np.sqrt((flow[..., 0] ** 2 + flow[..., 1] ** 2))
        median_flow = float(torch.from_numpy(flow_magnitude).median().item())
        mags.append(median_flow)
        prev = nxt
    mags = np.asarray(mags, dtype=np.float32)
    return mags

def select_indices_by_arc_length_abs(
    mags: np.ndarray,
    v_low: float = 0.4,      # px/帧，静止阈
    v_high: float = 4.0,     # px/帧，剧烈阈
    min_k: int = 4,
    max_k: int = 16,
    force_odd_gap: bool = False
):
    """
    基于光流弧长 S 等间隔采样关键帧（不归一化）。
    先用平均每帧运动量 m_bar 决定 K（线性插值到 [min_k,max_k]），再按 S 等间隔取点。
    """
    N = len(mags) + 1
    if N <= 2:
        return list(range(N))

    m_bar = float(mags.mean()) if len(mags) else 0.0
    if v_high <= v_low: v_high = v_low + 1e-6
    r = (m_bar - v_low) / (v_high - v_low)
    r = float(np.clip(r, 0.0, 1.0))
    K = int(round(min_k + r * (max_k - min_k)))
    K = int(np.clip(K, min_k, max_k))

    # 按弧长等距
    S = np.concatenate([[0.0], np.cumsum(mags)])  # 长度 N
    S_total = S[-1]
    if S_total <= 1e-6:
        # 几乎完全静止：均匀取时间点
        import pdb
        pdb.set_trace()
        return list(np.linspace(0, N-1, num=min(K, N), dtype=int))

    targets = np.linspace(0.0, S_total, num=min(K, N))
    idxs = [0]
    ptr = 1
    for t in targets[1:-1]:
        while ptr < N and S[ptr] < t:
            ptr += 1
        i = ptr
        if i < N and abs(S[i] - t) > abs(S[i-1] - t):
            i = i - 1
        idxs.append(int(i))
    idxs.append(N - 1)

    # 可选：强制相邻间隔为奇数，方便取整数中点
    if force_odd_gap and N >= 3 and len(idxs) >= 2:
        adj = [idxs[0]]
        for a, b in zip(idxs[:-1], idxs[1:]):
            if (b - a) % 2 == 0 and (b - a) >= 2:
                if b + 1 < N: b = b + 1
                elif a - 1 >= 0: a = a - 1
            if a <= adj[-1]: a = adj[-1]
            if b <= a: b = a + 1
            adj[-1] = a
            adj.append(b)
        clean = [adj[0]]
        for x in adj[1:]:
            if x > clean[-1]:
                clean.append(x)
        idxs = clean
    return idxs

def build_pairs_with_mid(idxs):
    """从关键帧构建 (i, j, c) 三元组，要求 j-i>=2，c 为整数中点。"""
    triples = []
    for i, j in zip(idxs[:-1], idxs[1:]):
        if j - i < 2:
            continue
        c = (i + j) // 2
        triples.append((i, j, c))
    return triples

@torch.no_grad()
def get_scene_consistency_v3(video_list, names = None):

    # --- NEW: 用已有的光流模型获取运动强度 ---
    global flow_model, flow_args
    init_flow_model()

    global dinov3
    init_dinov3_model()
    model, processor = dinov3
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    print('================= Start Scene Consistency (motion-downsampled) =================')
    scores = []
    for idx, frames in tqdm(enumerate(video_list)):
        # 运动驱动的关键帧抽取
        store_dir = os.path.join(names[idx], f'flow.pkl')
        if not os.path.exists(store_dir):
            os.makedirs(names[idx], exist_ok=True)
            mags = compute_motion_series(frames)
            print(f'store to: {store_dir}')
            with open(store_dir, 'wb') as f:
                # Pickle the dictionary and write it to the file
                pickle.dump(mags, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        else:
            # assert False
            with open(store_dir, 'rb') as f:
                mags = pickle.load(f)
        store_dir = os.path.join(names[idx], f'dino.pkl')
        if not os.path.exists(store_dir):
            feats = []
            for k in range(len(frames)):
                try:
                    img = load_image_hf(frames[k])
                except:
                    import pdb
                    pdb.set_trace()
                inputs = processor(images=img, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    out = model(**inputs)
                feats.append(out.pooler_output.squeeze(0))
            print(f'store to: {store_dir}')
            with open(store_dir, 'wb') as f:
                # Pickle the dictionary and write it to the file
                pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # assert False
            with open(store_dir, 'rb') as f:
                feats = pickle.load(f)
        
        idxs = select_indices_by_arc_length_abs(mags, v_low=1.0, v_high=10.0, min_k=3, max_k=20)
        if len(idxs) < 2:
            scores.append(0.0)
            continue
        # print(mags)
        print(idxs)

        feats_sel = []
        for k in idxs:
            feats_sel.append(feats[k])
        F = torch.stack(feats_sel)                   # M x D
        F1, F2 = F[0].unsqueeze(0).expand(len(F)-1, -1), F[1:]
        Fa, Fb = F[:-1], F[1:]

        # First→All 与 Adjacent 两路平均
        s1 = cos(F1, F2).clamp_min(0).mean()
        s2 = cos(Fa, Fb).clamp_min(0).mean()
        # scores.append(float((s1 + s2) * 0.5))
        scores.append(float(s2))

    score = float(np.mean(scores)) if scores else 0.0
    print(f'[motion-downsampled] dino consistency raw: {score}')
    return score