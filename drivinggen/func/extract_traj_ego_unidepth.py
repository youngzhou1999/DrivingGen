import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import numpy as np
import torch
from PIL import Image

import sys

# packs_path = os.path.abspath('third_parties')
# sys.path.append(packs_path)
from unidepth.models import UniDepthV1, UniDepthV2
from unidepth.utils import colorize, image_grid

import re
# sys.path.append('/mnt/cache/zhouyang/d_eval/VisualSLAM')
# from VO import VisualOdometry
from visual_slam.dataset import *
from visual_slam.vo import *
# from lib.visualization import plotting
import math

from ultralytics import YOLOv10

# from samurai.scripts.demo import samurai_main
import random
import numpy as np
from scipy.optimize import minimize
import torch
import json


depth_model=None
det_model=None

def init_depth_model():
    global depth_model
    print("Torch version:", torch.__version__)
    name = "unidepth-v2-vitl14"
    # model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
    depth_model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
    version = "v2"
    backbone = "vitl14"

    # depth_model = torch.hub.load(
    #     "/mnt/cache/zhouyang/d_eval/UniDepth",
    #     "UniDepth",
    #     version=version,
    #     backbone=backbone,
    #     pretrained=True,
    #     trust_repo=True,
    #     force_reload=True,
    #     source='local',
    # )

    # set resolution level (only V2)
    depth_model.resolution_level = 0

    # set interpolation mode (only V2)
    depth_model.interpolation_mode = "bilinear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = depth_model.to(device)

def init_det_model():
    global det_model
    det_model = YOLOv10('/shared_disk/users/yang.zhou/iclr_open_source/DrivingGen/ckpt/yolov10x.pt').cuda()

def set_task_list(base_path, local_rank, gt_json=None, model_name='gt', exp_id='free', all_id=8):
    runs = []

    if model_name == 'gt':
        with open(gt_json, 'r') as f:
            gt_json = json.load(f)
        dirs = gt_json
    else:
        dirs = os.listdir(base_path)
    if model_name == 'gt':
        # scenes = [os.path.join(base_path, f, model_name) for f in dirs]
        scenes = dirs
    else:
        # exp_id = exp_id + '-frames_25-conds_1-rounds_5'
        exp_id = exp_id
        scenes = [os.path.join(base_path, f, model_name, exp_id) for f in dirs]
        # assert len(scenes) == 222

    # 获取全局和本地rank
    global_rank = int(os.environ.get("RANK", 0))  # 默认值为0
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # 默认值为1

    import random
    random.seed(2026)
    random.shuffle(scenes)

    # 使用 numpy.array_split 来均匀分配数据
    data_chunks = np.array_split(scenes, world_size)  # 将数据分割为 world_size 份
    local_data_chunk = data_chunks[global_rank]  # 获取当前全局rank对应的部分数据

    # 再将每个机器的分割数据均分给8个GPU
    # local_data_chunks = np.array_split(local_data_chunk, 8)
    local_data_chunks = np.array_split(local_data_chunk, all_id)
    scenes = local_data_chunks[local_rank % all_id]  # 获取当前 local_rank 对应的数据
    
    for scene in scenes:
        if model_name == 'gt':
            runs.append(os.path.join(scene, 'CAM_F0'))
        else:
            runs.append(os.path.join(scene, 'images'))
    
    print(f"Total {len(runs)} data to process, local index {local_rank}")
    
    return runs

def reconstruct_global_trajectory(pixel_centers, depth_values, Ks, camera_poses, delta_d=None):
    # K_inv = np.linalg.inv(K)  # 内参矩阵逆
    global_trajectory = []

    for i, ((x_c, y_c), Z_c) in enumerate(zip(pixel_centers, depth_values)):
        # 1. 像素坐标 -> 相机坐标
        K_inv = np.linalg.inv(Ks[i])  # 内参矩阵逆
        pixel_coord = np.array([x_c, y_c, 1])
        if delta_d is not None:
            Z_c += delta_d[i].item()
        cam_coord = (K_inv @ pixel_coord) * Z_c

        # 2. 相机坐标 -> 全局坐标
        R, T = camera_poses[i]
        world_coord = R @ cam_coord + T

        global_trajectory.append(world_coord[[0,-1]]) # x, z
    
    return global_trajectory


def det_obj(frame):
    # one image
    global det_model
    result = det_model(frame)
    result = result[0]
    
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # result.save(filename=os.path.join(frame_out_dir, f"{i:05}.jpg"))  # save to disk
    '''
    {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
    78: 'hair drier', 79: 'toothbrush'}
    '''
    movable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    # try all first
    mask = (np.ones((frame.shape[0], frame.shape[1])) * 255).astype(np.uint8)
    person_car_pixel = []
    person_car_bbox = []    # xyxy
    if boxes:
        num_box = len(boxes)
        for id in range(num_box):   # shape: 6,
            # if int(boxes.cls[id].item()) == 9:
                # continue
            if int(boxes.cls[id].item()) in movable:
                xy = boxes.data[id, :4].cpu().numpy()
                x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                mask[y1:y2, x1:x2] = 0
                if int(boxes.cls[id].item()) in [0,1,2,3,5,6,7]:
                # if int(boxes.cls[id].item()) in [2]:
                    mid_x = int((x1+x2)/2)
                    mid_y = int((y1+y2)/2)
                    person_car_pixel.append([mid_x, mid_y])
                    person_car_bbox.append([x1,y1,x2,y2])
    return mask, person_car_pixel, person_car_bbox

def drive_roi_mask(h, w, keep=0.5, side=0.03):
    m = np.zeros((h,w), np.uint8)
    top, bot = 0, int(h*keep)
    l, r = int(w*side), int(w*(1-side))
    m[top:bot, l:r] = 255
    return m

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unidepth')
    
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--gt_meta_path', type=str, default='./vis_depth')
    parser.add_argument('--model_name', type=str, default='gt')
    parser.add_argument('--exp_id', type=str, default='free')
    parser.add_argument('--local_id', type=int, default=0)
    parser.add_argument('--all_id', type=int, default=8)
    args = parser.parse_args()

    init_depth_model()
    init_det_model()

    runs = set_task_list(args.root_path, int(args.local_id), args.gt_meta_path, args.model_name, args.exp_id, args.all_id)
    
    model_name = args.model_name
    exp_id = args.exp_id

    gt_json = args.gt_meta_path
    with open(gt_json, 'r') as f:
        gt_json = json.load(f)
    
    gt_paths = {}
    for gt_base in gt_json:
        # key = gt_base.split('/')[-2] + '+' + gt_base.split('/')[-1]
        gt_paths[gt_base] = gt_base

    # import pdb
    # pdb.set_trace()
    for run in runs:

        # if 'womd' not in run:
        #     print(f'skip womd: {run}')
        #     continue

        print(f'{args.local_id}: {run}')
    
        # video frames
        filenames = os.listdir(run)
        filenames = [os.path.join(run, f) for f in filenames]
        filenames.sort()

        rgbs = []
        depths_raw = []
        points_3d = []
        rgb_intrinsics = []
        masks = []
        depths_mov = []
        track_masks = []
        track_bboxs = []

        # frame_out_dir = os.path.join(args.outdir, 'frames')
        # os.system(f'rm -rf {frame_out_dir}')
        # os.makedirs(frame_out_dir, exist_ok=True)


        if model_name == 'gt':
            s_name = run.split('/')[-3] + '+' + run.split('/')[-2]
            log_base = os.path.join(args.outdir, s_name, model_name, 'unidepth')
        else:
            s_name = run.split('/')[-4]
            log_base = os.path.join(args.outdir, s_name, model_name, exp_id, 'unidepth')

        depth_out_dir = os.path.join(log_base, 'depth_frame')
        os.makedirs(depth_out_dir, exist_ok=True)

        # if os.path.exists(log_base+'-estimate_ego_traj_0619.pkl'):
        #     print(f'skip: {log_base}-estimate_ego_traj_0619.pkl')
        #     continue

        # mask_out_dir = os.path.join(args.outdir, 'masks')
        # os.system(f'rm -rf {mask_out_dir}')
        # os.makedirs(mask_out_dir, exist_ok=True)

        # seg_out_dir = os.path.join(args.outdir, 'segs')
        # os.system(f'rm -rf {seg_out_dir}')
        # os.makedirs(seg_out_dir, exist_ok=True)

        for k, filename in enumerate(filenames):
            if 'conds_20' in args.exp_id and k < 19: # the 20th as the first: 81 in total
                continue
            # print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
            rgb = np.array(Image.open(filename))

            mask, person_car, person_car_bbox = det_obj(rgb)
            mask_wh = drive_roi_mask(576, 1024)
            mask[mask_wh==0] = 0
            masks.append(mask)

            rgbs.append(rgb.copy())
            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
            # intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))

            # predict
            predictions = depth_model.infer(rgb_torch, None)
            depth = predictions["depth"].squeeze().cpu().numpy()
            depths_raw.append(depth.copy())

            # Intrinsics Prediction
            intrinsics = predictions["intrinsics"].squeeze(0).cpu().numpy()
            rgb_intrinsics.append(intrinsics)

            # Point Cloud in Camera Coordinate
            xyz = predictions["points"].cpu().squeeze(0).reshape(3, -1).t().numpy()
            points_3d.append(predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy().copy())

            # depth_pred_col = colorize(depth, vmin=0.01, vmax=10.0, cmap="magma_r")
            depth_pred_col = colorize(depth, cmap="jet")
            
            cv2.imwrite(os.path.join(depth_out_dir, f'{k:05}.png'), depth_pred_col)

        from moviepy.editor import ImageSequenceClip
        def images_to_video(image_folder, output_video, fps=30):
            images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if (img.endswith(".png") or img.endswith(".jpg"))]
            images.sort()  # Ensure the images are in the correct order
            # images.sort(key=lambda p: int(p.split('/')[-1].split('.')[0].split('_')[-1]))

            clip = ImageSequenceClip(images, fps=fps)
            clip.write_videofile(output_video, codec="libx264",
                verbose=False,
                logger=None

            )
        # images_to_video(frame_out_dir, os.path.join(args.outdir, f"frames.mp4"), fps=10)
        images_to_video(depth_out_dir, os.path.join(log_base, f"depths.mp4"), fps=10)

        # if not os.path.exists(log_base+'-estimate_ego_traj_0619.pkl'):
        if True:
            # if 'nuplan' in run:
            #     meta_path = '/mnt/cache/zhouyang/dg-bench/one-drive/nuplan_0822'
            # elif 'zod' in run:
            #     meta_path = '/mnt/cache/zhouyang/dg-bench/one-drive/zod_0822'
            # elif 'covla' in run:
            #     meta_path = '/mnt/cache/zhouyang/dg-bench/one-drive/covla_0822'
            # elif 'dojo' in run:
            #     meta_path = '/mnt/cache/zhouyang/dg-bench/one-drive/dojo_0822'
            # else:
            #     meta_path = None
            
            # 0924: bug in zod
            # if 'align' in args.root_path:
            #     gt_base = gt_paths[s_name]
            #     intrinsics_path = os.path.join(gt_base, 'intrinsic.npy')
            #     if 'conds_20' in args.exp_id:
            #         rgb_intrinsics = np.load(intrinsics_path, allow_pickle=True)[19:100]
            #     else:
            #         rgb_intrinsics = np.load(intrinsics_path, allow_pickle=True)

            dataset_handler = DatasetHandler(
                rgbs, depths_raw, rgb_intrinsics
            )

            check_data(dataset_handler, args.outdir)

            # Part 1. Features Extraction
            images = dataset_handler.images
            kp_list, des_list = extract_features_dataset(images, masks)


            # Part II. Feature Matching
            matches, matches2 = match_features_dataset(des_list)

            # print(dataset_handler.k)

            # Set to True if you want to use filtered matches or False otherwise
            is_main_filtered_m = True
            if is_main_filtered_m:
                # 0319: 0.6 seems not good
                dist_threshold = 0.7
                filtered_matches = filter_matches_dataset(matches, dist_threshold, matches2)
                matches = filtered_matches

            # Part III. Trajectory Estimation
            depth_maps = dataset_handler.depth_maps
            try:
                trajectory, poses_3x3 = estimate_trajectory(matches, kp_list, dataset_handler.k, depth_maps=depth_maps, dataset_handler=dataset_handler)
            except Exception as e:
                print(e)

                import pdb
                pdb.set_trace()

                print(f'extract ego fail for {run}')
                continue

            locs = []
            for i in range(0, trajectory.shape[1]):
                current_pos = trajectory[:, i]
                locs.append([current_pos.item(0), current_pos.item(2)])
            
            locs = np.array(locs, dtype=np.float32)
            # np.save(log_base+'-estimate_ego_traj.npy', locs)
            print(log_base, locs)
            ego_traj = {
                'locs': locs,
                'poses_3x3': poses_3x3,
            }
            import pickle
            with open(log_base+'-estimate_ego_traj.pkl', 'wb') as f:
                pickle.dump(ego_traj, f)
