import multiprocessing as mp
if __name__ == '__main__' or True:  # 或者直接 True
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import torch
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    # 旧版没有该函数时，返回 False 即可（与默认行为一致）
    torch.compiler.is_compiling = lambda: False

from cosmos_reason1_utils.script import init_script

init_script()

import argparse
import collections
import pathlib
import textwrap

import qwen_vl_utils
import transformers
import vllm
import yaml
from rich import print
from rich.pretty import pprint

from cosmos_reason1_utils.text import (
    PromptConfig,
    create_conversation,
    extract_tagged_text,
)
from cosmos_reason1_utils.vision import (
    VisionConfig,
    overlay_text_on_tensor,
    save_tensor,
)

ROOT = 'third_parties/cosmos-reason1'
SEPARATOR = "-" * 20

def pprint_dict(d: dict, name: str):
    """Pretty print a dictionary."""
    pprint(collections.namedtuple(name, d.keys())(**d), expand_all=True)

cosmos_r = None
def init_glm():
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print('mp error')
        pass  # 已经设置过
    # global model, processor
    # processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    # model = Glm4vForConditionalGeneration.from_pretrained(
    #     pretrained_model_name_or_path=MODEL_PATH,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    # )
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # Create model
    global cosmos_r

    prompt = "third_parties/cosmos-reason1/prompts/question.yaml"

    vision_config = f"{ROOT}/configs/vision_config.yaml"
    sampling_params = f"{ROOT}/configs/sampling_params.yaml"
    verbose = True

    # Load configs
    prompt_kwargs = yaml.safe_load(open(prompt, "rb"))
    prompt_config = PromptConfig.model_validate(prompt_kwargs)
    vision_kwargs = yaml.safe_load(open(vision_config, "rb"))
    _vision_config = VisionConfig.model_validate(vision_kwargs)
    sampling_kwargs = yaml.safe_load(open(sampling_params, "rb"))
    sampling_params = vllm.SamplingParams(**sampling_kwargs)
    if verbose:
        pprint_dict(vision_kwargs, "VisionConfig")
        pprint_dict(sampling_kwargs, "SamplingParams")

    # Create conversation
    system_prompts = [open(f"{ROOT}/prompts/addons/english.txt").read()]
    if prompt_config.system_prompt:
        system_prompts.append(prompt_config.system_prompt)
    if True and "<think>" not in prompt_config.system_prompt:
        if extract_tagged_text(prompt_config.system_prompt)[0]:
            raise ValueError(
                "Prompt already contains output format. Cannot add reasoning."
            )
        system_prompts.append(open(f"{ROOT}/prompts/addons/reasoning.txt").read())
    system_prompt = "\n\n".join(map(str.rstrip, system_prompts))

    print(SEPARATOR)
    print("System:")
    print(textwrap.indent(system_prompt.rstrip(), "  "))
    print(SEPARATOR)

    # Create model
    llm = vllm.LLM(
        model="nvidia/Cosmos-Reason1-7B",
        # revision=args.revision,
        limit_mm_per_prompt={"image": 120, "video": 1},
        enforce_eager=True,
        gpu_memory_utilization=0.6
    )

    # Process inputs
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained("nvidia/Cosmos-Reason1-7B")
    )

    cosmos_r = llm, processor, system_prompt, vision_kwargs, sampling_params


import re
def extract_answer(text: str):
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S|re.I)
    if not m:
        return None
    inner = re.sub(r"<[^>]+>", "", m.group(1))  # 先去标签
    m2 = re.search(r"\b(natural|unnatural)\b", inner, flags=re.I)
    return m2.group(1).lower() if m2 else None


# transformers original in drive_bench: 4.37.2
def glm_reasoning(gen_video_path, first_fids, first_box, last_fids, last_box, next_fids, glm_dir):
    os.makedirs(glm_dir, exist_ok=True)

    first_fid = first_fids[0] 
    
    first_img = cv2.imread(gen_video_path[first_fid])
    suffix = gen_video_path[first_fid][-4:]
    x1, y1, x2, y2 = first_box[0]
    cv2.rectangle(first_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(glm_dir, f'{first_fid:05d}{suffix}'), first_img)

    first_fid_2 = first_fids[1] 
    
    first_img_2 = cv2.imread(gen_video_path[first_fid_2])
    suffix = gen_video_path[first_fid_2][-4:]
    x1, y1, x2, y2 = first_box[1]
    cv2.rectangle(first_img_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(glm_dir, f'{first_fid_2:05d}{suffix}'), first_img_2)
    

    last_fid_2 = last_fids[0]

    last_img_2 = cv2.imread(gen_video_path[last_fid_2])
    suffix = gen_video_path[last_fid_2][-4:]
    x1, y1, x2, y2 = last_box[0]
    cv2.rectangle(last_img_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(glm_dir, f'{last_fid_2:05d}{suffix}'), last_img_2)

    last_fid = last_fids[-1]

    last_img = cv2.imread(gen_video_path[last_fid])
    suffix = gen_video_path[last_fid][-4:]
    x1, y1, x2, y2 = last_box[1]
    cv2.rectangle(last_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(glm_dir, f'{last_fid:05d}{suffix}'), last_img)

    next_fid = next_fids[0]

    next_img = cv2.imread(gen_video_path[next_fid])
    suffix = gen_video_path[next_fid][-4:]
    cv2.imwrite(os.path.join(glm_dir, f'{next_fid:05d}{suffix}'), next_img)

    next_fid_2 = next_fids[1]

    next_img_2 = cv2.imread(gen_video_path[next_fid_2])
    suffix = gen_video_path[next_fid_2][-4:]
    cv2.imwrite(os.path.join(glm_dir, f'{next_fid_2:05d}{suffix}'), next_img_2)




    global cosmos_r
    if cosmos_r is None:
        init_glm()
    llm, processor, system_prompt, vision_kwargs, sampling_params = cosmos_r


    # question = '''Given four frames (first appearance in I1, last appearance in I2, then disappear in I3-I4) of the same green-boxed object, \
    #             decide Natural vs Unnatural disappearance. Use visual continuity, motion continuity, and interactions with surrounding vehicles/environment. \
    #             If the object is missing or visibly cut already in I3 or I4, mark Unnatural. Output: label: Natural|Unnatural and a brief reason referencing I1–I6.'''

    question = '''Given frames around the moment the same green-boxed object disappears, classify the disappearance as Natural (e.g., occlusion, leaving the field of view) or Unnatural (e.g., abrupt/non-physical disappearance). \
        Base your decision on visual continuity, motion continuity, and the object’s interactions with surrounding vehicles and the environment.'''

    images = [
        os.path.join(glm_dir, f'{first_fid:05d}{suffix}'),
        os.path.join(glm_dir, f'{first_fid_2:05d}{suffix}'),
        os.path.join(glm_dir, f'{last_fid_2:05d}{suffix}'),
        os.path.join(glm_dir, f'{last_fid:05d}{suffix}'),
        os.path.join(glm_dir, f'{next_fid:05d}{suffix}'),
        os.path.join(glm_dir, f'{next_fid_2:05d}{suffix}')
    ]

    videos = []

    user_prompt = question
    if not user_prompt:
        raise ValueError("No user prompt provided.")
    user_prompt = user_prompt.rstrip()
    conversation = create_conversation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        videos=videos,
        vision_kwargs=vision_kwargs,
    )
    # pprint(conversation, expand_all=True)
    # print(SEPARATOR)
    # print("User:")
    # print(textwrap.indent(user_prompt.rstrip(), "  "))
    # print(SEPARATOR)


    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation, return_video_kwargs=True
    )

    # Run inference
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    print(SEPARATOR)
    for output in outputs[0].outputs:
        output_text = output.text
        print("Assistant:")
        print(textwrap.indent(output_text.rstrip(), "  "))
    print(SEPARATOR)

    result, _ = extract_tagged_text(output_text)
    if result:
        pprint_dict(result, "Result")

    if 'Unnatural' in result['answer'][0]:
        return False
    return True

from typing import Sequence, List, Optional, TypeVar

T = TypeVar("T")

def sample_track(
    seq: Sequence[T],
    keep_last: int = 10,
    total_keep: Optional[int] = None,   # 方式A：目标总数（含最后10帧）
    stride: Optional[int] = None,       # 方式B：步长采样（对前段）
) -> List[T]:
    """
    下采样一条轨迹序列 seq（如帧/box 列表）：
    - 始终保留首帧（index=0）与最后 keep_last 帧（若长度不足则全保留）
    - 其余部分按 total_keep 或 stride 等距抽样
    - 返回与输入同顺序的子序列（去重）

    二选一传参：
      * total_keep：希望最终保留的总帧数上限（含最后 keep_last），例如 32
      * stride：对“前段（不含最后 keep_last）”按固定步长抽样，例如 10
    """
    n = len(seq)
    if n == 0:
        return []

    # 长度不超过保底：全保留
    if n <= keep_last + 1:
        # 这里已包含首帧与尾部
        return list(seq)

    # 需要抽样的“前段”范围：[0, n-keep_last) —— 注意首帧必须保留
    front_len = n - keep_last
    early_indices: List[int] = []

    if total_keep is not None and stride is not None:
        raise ValueError("Provide either total_keep or stride, not both.")
    if total_keep is None and stride is None:
        raise ValueError("You must provide total_keep or stride.")

    if total_keep is not None:
        # 方式A：控制最终总数
        if total_keep >= n:
            # 目标数不小于原长度，直接全保留
            early_indices = list(range(front_len))  # 前段全留
        else:
            # 需要从前段选 early_budget 个点，包含首帧索引 0
            early_budget = max(total_keep - keep_last, 1)
            if early_budget == 1:
                early_indices = [0]
            else:
                # 等距取样（含 0 和 front_len-1），再把 front_len-1 去掉避免与尾段重叠
                # 用四舍五入均匀取整
                early_indices = [
                    round(i * (front_len - 1) / (early_budget - 1))
                    for i in range(early_budget)
                ]
                # 去掉 front_len-1，避免与“最后 keep_last 帧”重复
                if early_indices and early_indices[-1] == front_len - 1:
                    early_indices.pop()
    else:
        # 方式B：固定步长，对前段按 stride 取样，并确保包含首帧
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")
        early_indices = list(range(0, front_len, stride))
        if early_indices[0] != 0:
            early_indices.insert(0, 0)
        # 避免包含 front_len-1（会与尾段重叠）
        if early_indices and early_indices[-1] == front_len - 1:
            early_indices.pop()

    # 尾段索引：最后 keep_last 帧（一定包含 n-1）
    tail_indices = list(range(n - keep_last, n))

    # 合并去重并保持顺序
    seen = set()
    keep_indices: List[int] = []
    for idx in early_indices + tail_indices:
        if idx not in seen:
            seen.add(idx)
            keep_indices.append(idx)

    # 组装输出
    return [seq[i] for i in keep_indices]


def glm_reasoning_2(gen_video_path, track_boxes, next_fids, glm_dir):
    os.makedirs(glm_dir, exist_ok=True)

    images = []
    track_boxes = sample_track(track_boxes, keep_last=3, total_keep=10)

    for first_fid, box in track_boxes: 
        
        first_img = cv2.imread(gen_video_path[first_fid])
        suffix = gen_video_path[first_fid][-4:]
        x1, y1, x2, y2 = box
        cv2.rectangle(first_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(glm_dir, f'{first_fid:05d}{suffix}'), first_img)
        images.append(os.path.join(glm_dir, f'{first_fid:05d}{suffix}'))

    next_fid = next_fids[0]

    next_img = cv2.imread(gen_video_path[next_fid])
    suffix = gen_video_path[next_fid][-4:]
    cv2.imwrite(os.path.join(glm_dir, f'{next_fid:05d}{suffix}'), next_img)

    next_fid_2 = next_fids[1]

    next_img_2 = cv2.imread(gen_video_path[next_fid_2])
    suffix = gen_video_path[next_fid_2][-4:]
    cv2.imwrite(os.path.join(glm_dir, f'{next_fid_2:05d}{suffix}'), next_img_2)



    global cosmos_r
    if cosmos_r is None:
        init_glm()
    llm, processor, system_prompt, vision_kwargs, sampling_params = cosmos_r


    # question = '''Given four frames (first appearance in I1, last appearance in I2, then disappear in I3-I4) of the same green-boxed object, \
    #             decide Natural vs Unnatural disappearance. Use visual continuity, motion continuity, and interactions with surrounding vehicles/environment. \
    #             If the object is missing or visibly cut already in I3 or I4, mark Unnatural. Output: label: Natural|Unnatural and a brief reason referencing I1–I6.'''

    question = '''Given frames around the moment the same green-boxed object disappears, classify the disappearance as Natural (e.g., occlusion, leaving the field of view) or Unnatural (abrupt/non-physical disappearance). \
        Base your decision on visual continuity, motion continuity, and the object’s interactions with surrounding vehicles and the environment.'''

    images += [
        # os.path.join(glm_dir, f'{first_fid:05d}{suffix}'),
        # os.path.join(glm_dir, f'{first_fid_2:05d}{suffix}'),
        # os.path.join(glm_dir, f'{last_fid_2:05d}{suffix}'),
        # os.path.join(glm_dir, f'{last_fid:05d}{suffix}'),
        os.path.join(glm_dir, f'{next_fid:05d}{suffix}'),
        os.path.join(glm_dir, f'{next_fid_2:05d}{suffix}')
    ]

    videos = []

    user_prompt = question
    if not user_prompt:
        raise ValueError("No user prompt provided.")
    user_prompt = user_prompt.rstrip()
    conversation = create_conversation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        videos=videos,
        vision_kwargs=vision_kwargs,
    )
    # pprint(conversation, expand_all=True)
    # print(SEPARATOR)
    # print("User:")
    # print(textwrap.indent(user_prompt.rstrip(), "  "))
    # print(SEPARATOR)


    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation, return_video_kwargs=True
    )

    # Run inference
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    print(SEPARATOR)
    for output in outputs[0].outputs:
        output_text = output.text
        print("Assistant:")
        print(textwrap.indent(output_text.rstrip(), "  "))
    print(SEPARATOR)

    result, _ = extract_tagged_text(output_text)
    if result:
        pprint_dict(result, "Result")

    if 'Unnatural' in result['answer'][0]:
        return False
    return True



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

# from objects.missing_and_occ import (
#     disappeared_suddenly, get_missing_per_scene, get_missing_per_agent
# )

import numpy as np
from typing import List, Tuple, Dict, Sequence

# --------- 基础工具 ---------
def bbox_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    单个 box 与一组 boxes 的 IoU
    box.shape == (4,)     ->  [x1, y1, x2, y2]
    boxes.shape == (N, 4)
    """
    # 交集
    tl = np.maximum(box[:2], boxes[:, :2])            # top-left
    br = np.minimum(box[2:], boxes[:, 2:])            # bottom-right
    inter_wh = np.clip(br - tl, a_min=0, a_max=None)
    inter = inter_wh[:, 0] * inter_wh[:, 1]

    # 面积
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter + 1e-9
    return inter / union


def near_image_edge(box: Sequence[float],
                    img_wh: Tuple[int, int],
                    margin_ratio: float = 0.05) -> bool:
    """
    判断 bbox 是否靠近图像边缘（留出 margin_ratio 的相对边距）
    """
    w, h = img_wh
    x1, y1, x2, y2 = box
    margin_x = w * margin_ratio
    margin_y = h * margin_ratio
    return (
        x1 <= margin_x or
        # y1 <= margin_y or
        (w - x2) <= margin_x or
        (h - y2) <= margin_y
    )


# from .glm import glm_reasoning, glm_reasoning_2
# --------- 主函数 ---------
def disappeared_suddenly(
    track_boxes: List[Tuple[int, Sequence[float]]],
    other_boxes_by_frame: Dict[int, List[Sequence[float]]],
    gen_video_path,
    glm_dir,
    img_size: Tuple[int, int],
    *,
    edge_margin: float = 0.05,
    iou_threshold: float = 0.2,
    min_track_len: int = 5
) -> bool:
    """
    判断给定物体轨迹是否“突然消失”。
    ----------
    Args
    ----
    track_boxes            : [(frame_id, [x1,y1,x2,y2]), ...] 按帧排序
    other_boxes_by_frame   : {frame_id: [[x1,y1,x2,y2], ...], ...}
                             - 不包含当前 track 的 bbox（多目标检测 / 其他 track）
    img_size               : (W, H)  视频分辨率
    edge_margin            : 判定“靠边”的宽度占比
    iou_threshold          : 与同帧其他目标的最大 IoU 上限
    min_track_len          : 轨迹太短时直接忽略
    Returns
    -------
    True  -> 满足“突然消失”条件  
    False -> 未满足
    """
    if len(track_boxes) < min_track_len:
        print(f'len track_bbox is {len(track_boxes)}: too short, skip')
        return False                      # 轨迹太短，稳定性不足

    # 1) 最后出现的帧
    last_fid, last_box = track_boxes[-1]
    last_box = np.asarray(last_box, dtype=float)

    if last_fid == 100:
        print(f'no missing: 100 frames')
        return False                      # 最后一帧不算“突然消失”，因为没有后续帧

    # 2) 检查是否靠近边缘
    # if near_image_edge(last_box, img_size, edge_margin):
    #     print(f'no missing: in corner')
    #     return False                      # 合理：对象跑出画面

    # 3) 与同帧其他 bbox 的 IoU
    # others = other_boxes_by_frame.get(last_fid, [])
    # if len(others):
    #     ious = bbox_iou(last_box, np.asarray(others, dtype=float))
    #     if (ious >= iou_threshold).any():
    #         print(f'no missing: occ with objects from the first frame')
    #         return False                  # 可能只是 track ID 断了 / 与别人重叠

    # 4) normal ones
    first_fid, first_box = track_boxes[0]
    first_fid_, first_box_ = track_boxes[1]
    first_fids = [first_fid, first_fid_]
    first_boxs = [first_box, first_box_]

    last_fids = []
    last_boxs = []
    idx_last = max(0, len(track_boxes)-5)
    for id in [idx_last, -1]:
        last_fid, last_box = track_boxes[id]
        last_fids.append(last_fid)
        last_boxs.append(last_box)

    next_fids = [last_fids[-1]+1]
    if last_fids[-1]+2 <=100:
        next_fids.append(last_fids[-1]+2)
    else:
        next_fids.append(last_fids[-1]+1)

    is_normal_missing = glm_reasoning(
        gen_video_path,
        first_fids, first_boxs,
        last_fids, last_boxs,
        next_fids,
        glm_dir)
    # is_normal_missing = glm_reasoning_2(
    #     gen_video_path,
    #     track_boxes,
    #     next_fids,
    #     glm_dir)
    if is_normal_missing:
        return False
    
    # 若满足上述所有条件，则判定为“突然消失”
    return True


def get_missing_per_scene(missings):
    scene_score = []
    count = 0
    for id, missing in missings:
        if missing:
            # score = 1.0 - ( (id+1) / 101)
            score = 1
            count += 1
            scene_score.append(score)
    if len(scene_score) > 0:
        # return np.mean(scene_score) * count / len(missings)
        return np.mean(scene_score)
    else:
        return 0.0
    

def get_missing_per_agent(missings):
    scene_score = []
    count = 0
    for id, missing in missings:
        if missing:
            # score = 1.0 - ( (id+1) / 101)
            score = 1
            count += 1
    scene_score = count
    return scene_score
    if len(scene_score) > 0:
        # return np.mean(scene_score) * count / len(missings)
        return np.mean(scene_score)
    else:
        return 0.0





# ---------- 场景 1：单帧遮挡 ----------
def occluded_in_frame(
    box: Sequence[float],
    other_boxes: List[Sequence[float]],
    img_wh: Tuple[int, int],
    *,
    edge_margin: float = 0.05,
    iou_thr: float = 0.5
) -> bool:
    """
    判断单帧内 box 是否被“遮挡”  
    条件：不靠边 & 与任意 other box 的 IoU ≥ iou_thr
    """
    if near_image_edge(box, img_wh, edge_margin) or not other_boxes:
        return False

    ious = bbox_iou(np.asarray(box, float), np.asarray(other_boxes, float))
    return bool((ious >= iou_thr).any())


# ---------- 场景 2：整条 track 的遮挡统计 ----------
def track_occlusion_score(
    track_boxes: List[Tuple[int, Sequence[float]]],
    other_boxes_by_frame: Dict[int, List[Sequence[float]]],
    img_wh: Tuple[int, int],
    *,
    edge_margin: float = 0.05,
    iou_thr: float = 0.5,
) -> Tuple[int, float, List[int]]:
    """
    给定一条轨迹，统计其被遮挡情况
    Returns
    -------
    n_occ   : 被遮挡的帧数
    ratio   : n_occ / len(track_boxes)
    occ_fids: 被遮挡帧列表
    """
    occ_fids = [
        fid for fid, box in track_boxes
        if occluded_in_frame(
            box,
            other_boxes_by_frame.get(fid, []),
            img_wh,
            edge_margin=edge_margin,
            iou_thr=iou_thr
        )
    ]
    n_occ = len(occ_fids)
    ratio = n_occ / len(track_boxes) if track_boxes else 0.0
    return n_occ, ratio, occ_fids


import torch
import pickle
from numpy.typing import ArrayLike

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


def get_agent_missing(valid_agents_runs, agents_bbox, agents_label, names, img_dirs):
    missing_rate = []
    num_agent = 0
    num_missing_agent = 0

    print(f'len valid runs: {len(valid_agents_runs)}, len agents_bbox: {len(agents_bbox)}')
    for scene_id, scene_agents in enumerate(agents_bbox):
        is_missing = []
        num_agent += len(scene_agents)

        gen_video_path = [os.path.join(img_dirs[scene_id], f) for f in os.listdir(img_dirs[scene_id])]
        sorted(gen_video_path)
        glm_dir = names[scene_id]
        
        video_img_dict = {}
        for img_id, img in enumerate(gen_video_path):
            video_img_dict[img_id] = img

        for agent_id, scene_agent in enumerate(scene_agents):

            other_agents = scene_agents.copy()
            other_agents.remove(scene_agent)
            other_boxes_by_frame = {}
            for other in other_agents:
                for o in other:
                    f_id, bbox = o
                    if f_id not in other_boxes_by_frame:
                        other_boxes_by_frame[f_id] = [bbox]
                    else:
                        other_boxes_by_frame[f_id].append(bbox)

            glm_this = os.path.join(glm_dir, f'{agent_id}')
            missing = disappeared_suddenly(
                scene_agent,
                other_boxes_by_frame,
                video_img_dict,
                glm_this,
                img_size=(1024, 576),
                edge_margin=0.1,
                iou_threshold=0.5,
                min_track_len=2
            )
            # print(scene_agent)
            print(f'{names[scene_id]} - agent {agent_id} missing: {missing}')
            is_missing.append((scene_agent, missing))
        missing_rate.append((get_missing_per_scene(is_missing), is_missing))
        num_missing_agent += get_missing_per_agent(is_missing)

    assert len(missing_rate) == len(valid_agents_runs)

    missing_rate = [m[0] for m in missing_rate]
    print(len(missing_rate))
    missing_rate = np.nanmean(missing_rate, axis=0)
    return 1-missing_rate
    # agent_quality['agent_quality']['object_missing_error'] = missing_rate
    metrics['objects-feasibility'] = {}
    metrics['objects-feasibility']['object_missing_error_scene'] = 1-missing_rate
    metrics['objects-feasibility']['object_missing_error_agent'] = num_missing_agent / num_agent

