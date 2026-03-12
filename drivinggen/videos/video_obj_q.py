import torch
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2
from tqdm import tqdm
from .p2020_v2 import single_frame_metrics as single_frame_metrics_v2
from .p2020_v2 import video_metrics as video_metrics_v2

def _process_one_video(images):
    """单个视频：读取→video级指标→逐帧指标均值，返回两个dict"""
    images_np = [cv2.imread(f) for f in images]

    # 原来的整段指标
    adj_metric = video_metrics_v2(images_np)

    # 原来的逐帧 → 均值
    this_dict = defaultdict(list)
    for image in images_np:
        img_metric = single_frame_metrics_v2(image)
        for k, v in img_metric.items():
            this_dict[k].append(v)
    frame_mean = {k: float(np.nanmean(np.asarray(v))) for k, v in this_dict.items()}

    return adj_metric, frame_mean
@torch.no_grad()
def get_objective_quality_v2(video_list):

    # global objective_quality_gt
    
    # val_dict = {}
    # video_val_dict = {}
    # print(f'=========================Start Objective Quality=========================')
    # for images in tqdm(video_list):
    #     images_np = [cv2.imread(f) for f in images]
    #     adj_metric = video_metrics_v2(images_np)
    #     for k, v in adj_metric.items():
    #         if k in video_val_dict:
    #             video_val_dict[k].append(v)
    #         else:
    #             video_val_dict[k] = [v]
    #     this_dict = {}
    #     for image in images_np:
    #         img_metric = single_frame_metrics_v2(image)
    #         for k, v in img_metric.items():
    #             if k in this_dict:
    #                 this_dict[k].append(v)
    #             else:
    #                 this_dict[k] = [v]
    #     for k, v in this_dict.items():
    #         v = np.nanmean(np.array(v))
    #         if k in val_dict:
    #             val_dict[k].append(v)
    #         else:
    #             val_dict[k] = [v]


    val_dict = defaultdict(list)
    video_val_dict = defaultdict(list)
    print('=========================Start Objective Quality=========================')

    max_workers = max(1, (os.cpu_count() or 2) - 1)  # 也可手动指定

    print(f'using {max_workers} worker')

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for adj_metric, frame_mean in tqdm(ex.map(_process_one_video, video_list),
                                        total=len(video_list)):
            for k, v in adj_metric.items():
                video_val_dict[k].append(v)
            for k, v in frame_mean.items():
                val_dict[k].append(v)

    # 如果需要纯 dict：
    val_dict = dict(val_dict)
    video_val_dict = dict(video_val_dict)

    # if not os.path.exists('./objective_quality_5_95.pkl'):
    #     normed_stat_gt = {}
    #     for k,v in val_dict.items():
    #         mu, sigma, z_min, z_max = zscore_rescale_gt(v, q_low=5, q_high=95)
    #         normed_stat_gt[k] = [mu, sigma, z_min, z_max]
    #     import pdb
    #     pdb.set_trace()
    #     for k,v in video_val_dict.items():
    #         mu, sigma, z_min, z_max = zscore_rescale_gt(v, q_low=5, q_high=95)
    #         normed_stat_gt[k] = [mu, sigma, z_min, z_max]
    #     with open('./objective_quality_5_95.pkl', 'wb') as f:
    #         pickle.dump(normed_stat_gt, f)
    # else:
    #     with open('./objective_quality_5_95.pkl', 'rb') as f:
    #         objective_quality_gt = pickle.load(f)

    normed_score_infer = {}
    for k, v in val_dict.items():
        # score = objective_quality_zscore_rescale_infer(v, key=k)
        score = np.nanmean(np.array(v))
        if 'fmp_alias' in k:
            score = 1 - score
        normed_score_infer[k] = score
        print(f'{k} score mean: {np.nanmean(score)}')

    for k, v in video_val_dict.items():
        # score = objective_quality_zscore_rescale_infer(v, key=k)
        score = np.nanmean(np.array(v))
        log = np.array(v)
        if 'fmp_alias' in k:
            score = 1 - score
        normed_score_infer[k] = score
        print(f'video {k} score mean: {np.nanmean(score)}')

    
    # iaq_per_img = iaq_per_sample(normed_score_infer)

    # iaq_final = iaq_per_img.mean()
    # gt 0.53
    # log = np.array([v for k, v in normed_score_infer.items()])
    avg = np.mean(np.array([v for k, v in normed_score_infer.items()]))
    normed_score_infer['avg'] = avg

    print(normed_score_infer)
    return float(normed_score_infer['mmp_alias'])
    return normed_score_infer, log

    return iaq_final