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

# sys.path.append('/shared_disk/users/yang.zhou/iclr_open_source/samurai')
sys.path.append(os.path.abspath('third_parties'))
from samurai.scripts.demo import samurai_main
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
    person_car_label = {}
    lookup_dict = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    if boxes:
        num_box = len(boxes)
        for id in range(num_box):   # shape: 6,
            # if int(boxes.cls[id].item()) == 9:
                # continue
            if int(boxes.cls[id].item()) in movable:
                xy = boxes.data[id, :4].cpu().numpy()
                x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                mask[y1:y2, x1:x2] = 0
                if int(boxes.cls[id].item()) in [0, 1, 2, 3, 5, 7] and boxes.conf[id].item() >= 0.3:
                # if int(boxes.cls[id].item()) in [2]:
                    mid_x = int((x1+x2)/2)
                    mid_y = int((y1+y2)/2)
                    person_car_pixel.append([mid_x, mid_y])
                    person_car_bbox.append([x1,y1,x2,y2])
                    person_car_label[f'{x1}-{y1}-{x2}-{y2}'] = lookup_dict[int(boxes.cls[id].item())]
    return mask, person_car_pixel, person_car_bbox, person_car_label


def estimate_depth_from_mask(depth, mask, method='median_top', use_percentile=False):
    """
    给定目标的 mask 和 depth 图，返回更鲁棒的深度估计（单位：米）

    参数:
    - depth: (H, W) float，深度图
    - mask: (H, W) bool，目标掩码
    - method: 'median_top', 'mean_top', 'full_median' 等
    - use_percentile: 是否用 10%-90% 截断平均来增强稳定性

    返回:
    - depth_est: float，估计深度值，单位米；若失败返回 None
    """
    if not np.any(mask):
        return None

    # 根据方法截取 mask 区域
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    h = y_max - y_min + 1

    # seems wild
    if 'top' in method:
        threshold = y_min + h * 0.5
        top_mask = mask.copy()
        for y, x in zip(ys, xs):
            if y >= threshold:
                top_mask[y, x] = False
        target_mask = top_mask
    else:
        target_mask = mask


    valid = depth[target_mask > 0]

    if len(valid) == 0:
        return None

    if use_percentile:
        # 去掉极端值再平均
        lower, upper = np.percentile(valid, [2, 98])
        valid = valid[(valid >= lower) & (valid <= upper)]

    if 'mean' in method:
        return float(np.mean(valid))
    else:
        return float(np.median(valid))


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
    for run in runs:

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

        # if os.path.exists(log_base+'-estimate_agents_traj.pkl') and \
        #  os.path.exists(log_base+'-estimate_agents_bbox.pkl') and \
        #  os.path.exists(log_base+'-estimate_agents_bbox_label.pkl'):
        #     print(f'skip: {log_base}-estimate_agents_traj.pkl')
        #     continue


        # mask_out_dir = os.path.join(args.outdir, 'masks')
        # os.system(f'rm -rf {mask_out_dir}')
        # os.makedirs(mask_out_dir, exist_ok=True)

        seg_out_dir = os.path.join(log_base, 'track_frame')
        os.makedirs(seg_out_dir, exist_ok=True)

        for k, filename in enumerate(filenames):
            if 'conds_20' in args.exp_id and k < 19: # the 20th as the first: 81 in total
                continue
            # print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
            rgb = np.array(Image.open(filename))

            if k == 0:
                mask, person_car, person_car_bbox, person_car_label = det_obj(rgb)
                # debug
                # if len(person_car) == 0:
                #     print(f'No person/car detected in {filename}, skip')
                #     break
            else:
                mask, _, _, _ = det_obj(rgb)
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
            
            # cv2.imwrite(os.path.join(depth_out_dir, f'{k:05}.png'), depth_pred_col)

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
        # images_to_video(depth_out_dir, os.path.join(log_base, f"depths.mp4"), fps=10)

        # debug
        # if len(person_car_bbox) == 0:
        #     # print(f'No person/car detected in {run}, skip')
        #     continue
        if (not os.path.exists(log_base+'-estimate_ego_traj.pkl')):
            # ego estimate
            import pdb
            pdb.set_trace()
            meta_path = '/mnt/cache/zhouyang/dg-bench/nuplan_1.1/val_sensor_data_10hz_0530'
            s_name = s_name.split('+')
            intrinsics_path = os.path.join(meta_path, s_name[0] + '+' + s_name[1], s_name[2], 'intrinsic.npy')
            if 'conds_20' in args.exp_id:
                rgb_intrinsics = np.load(intrinsics_path, allow_pickle=True)[19:100]
            else:
                rgb_intrinsics = np.load(intrinsics_path, allow_pickle=True)

            dataset_handler = DatasetHandler(
                rgbs, depths_raw, rgb_intrinsics
            )

            check_data(dataset_handler, args.outdir)

            # Part 1. Features Extraction
            images = dataset_handler.images
            kp_list, des_list = extract_features_dataset(images, masks)


            # Part II. Feature Matching
            matches, matches2 = match_features_dataset(des_list)

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

                print(f'extract ego fail for {run}')
                continue

            locs = []
            for i in range(0, trajectory.shape[1]):
                current_pos = trajectory[:, i]
                locs.append([current_pos.item(0), current_pos.item(2)])
            
            locs = np.array(locs, dtype=np.float32)
            # np.save(log_base+'-estimate_ego_traj-0611.npy', locs)
            # np.save(log_base+'-estimate_ego_traj.npy', locs)
            ego_traj = {
                'locs': locs,
                'poses_3x3': poses_3x3,
            }
            import pickle
            with open(log_base+'-estimate_ego_traj.pkl', 'wb') as f:
                pickle.dump(ego_traj, f)
        else:
            print('loading existing ego')
            import pickle
            with open(log_base+'-estimate_ego_traj.pkl', 'rb') as f:
                ego_traj = pickle.load(f)
            locs = ego_traj['locs']
            poses_3x3 = ego_traj['poses_3x3']

        # track
        mov_track_bbox= []
        mov_track_mask = []
        mov_track_label = []
        if len(person_car_bbox) > 0:
            import seaborn as sns
            import matplotlib.colors as mcolors

            def sns_to_cv_colors(n: int, palette="husl"):
                rgb = sns.color_palette(palette, n)        # 0-1 RGB
                return [(int(b*255), int(g*255), int(r*255)) for r, g, b in rgb]

            colors = sns_to_cv_colors(120, "husl")      # HUSL 均匀分布 120 色

            person_car_bbox_sorted = sorted(
                person_car_bbox,
                key=lambda b: (
                    (b[0] + b[2]) / 2,   # ① 中心 x —— 左→右
                    (b[1] + b[3]) / 2    # ② 中心 y —— 上→下（图像坐标 y 向下增）
                )
            )
            for box_id, bbox in enumerate(person_car_bbox_sorted):
                mov_track_label.append(person_car_label[f'{bbox[0]}-{bbox[1]}-{bbox[2]}-{bbox[3]}'])
                track_mask, track_bbox = samurai_main(
                    video_path=filenames,
                    txt_path=bbox,
                    save_to_video=False,
                    seg_out_dir=seg_out_dir,
                )
                # update save video
                if box_id > 0:
                    img_dir = seg_out_dir
                else:
                    img_dir = run
                boxes_dict = {}
                for f_id, bbox in track_bbox:
                    if f_id not in boxes_dict:
                        boxes_dict[f_id] = [bbox]
                    else:
                        boxes_dict[f_id].append(bbox)
                for fn in os.listdir(img_dir):
                    img = cv2.imread(os.path.join(img_dir, fn))
                    
                    # 取出该帧全部 bbox 并绘制
                    for (x1, y1, x2, y2) in boxes_dict.get(int(fn.split('.')[0].split('_')[0]), []):
                        cv2.rectangle(img, (x1, y1), (x2, y2), colors[box_id], 2)

                    # 保存图片
                    cv2.imwrite(os.path.join(seg_out_dir, fn), img)

                mov_track_mask.append(track_mask)
                mov_track_bbox.append(track_bbox)
            images_to_video(seg_out_dir, os.path.join(log_base, f"track.mp4"), fps=10)
        if len(mov_track_bbox) > 0:
            mov_pixel_frames = []
            mov_depth_centers = []
            mov_ids = []
            for mov_id, mov_box_i in enumerate(mov_track_bbox):
                this_pixel_frames = []
                this_depth_centers = []
                mov_mask_i = mov_track_mask[mov_id]
                mov_ids.append([m[0] for m in mov_box_i])
                for (frame_id, bbox) in mov_box_i:
                    x1, y1, x2, y2 = bbox
                    pc_x = int((x1+x2)/2)
                    pc_y = int((y1+y2)/2)
                    mask_i = (mov_mask_i[frame_id] > 0)
                    assert mask_i.sum() > 0
                    pc_d = estimate_depth_from_mask(depths_raw[frame_id], mov_mask_i[frame_id], method='mean', use_percentile=True)
                    this_pixel_frames.append((pc_x, pc_y))
                    this_depth_centers.append(pc_d)
                mov_pixel_frames.append(this_pixel_frames)
                mov_depth_centers.append(this_depth_centers)

            agents_traj = []
            for id_p, (pixel_values, depth_values) in enumerate(zip(mov_pixel_frames, mov_depth_centers)):
                mov_label = mov_track_label[id_p]
                ids_p = mov_ids[id_p]
                this_intrinsics = [rgb_intrinsics[id] for id in ids_p]
                this_poses = [poses_3x3[id] for id in ids_p]
                focal_global_coords = reconstruct_global_trajectory(pixel_values, depth_values, this_intrinsics, this_poses)
                agents_traj.append((ids_p, focal_global_coords, mov_label))

            # ### labels
            # import pdb
            # pdb.set_trace()
            # mov_track_labels = []
            # for mov_id, mov_box_i in enumerate(mov_track_bbox):
            #     this_mov_labels = []
            #     for frame, agent_bbox in mov_box_i:
            #         lable_this_frame = person_car_labels[frame]
            #         bboxes = list(lable_this_frame.keys())
            #         candidate_box = []
            #         for bbox in bboxes:
            #             c_x1, c_y1, c_x2, c_y2 = bbox.split('-')
            #             c_x1 = int(c_x1)
            #             c_y1 = int(c_y1)
            #             c_x2 = int(c_x2)
            #             c_y2 = int(c_y2)
            #             candidate_box.append([c_x1, c_y1, c_x2, c_y2])
            #         candidate_box = np.array(candidate_box).astype(np.int32)
            #         agent_bbox = np.array(agent_bbox).astype(np.int32)

            #         import pdb
            #         pdb.set_trace()

            #         match_box , _ = max_iou_box(agent_bbox, candidate_box)
            #         match_box = match_box.astype(np.int32)
            #         key = f'{match_box[0]}-{match_box[1]}-{match_box[2]}-{match_box[3]}'
            #         label = lable_this_frame[key]
            #         mov_track_labels.append((frame, label))
            #     mov_track_labels.append((mov_id, this_mov_labels))
            
            # save to pickle
            import pickle
            with open(log_base+'-estimate_agents_traj.pkl', 'wb') as f:
                pickle.dump(agents_traj, f)
            # np.save(log_base+'-estimate_agents_traj.npy', agents_traj)
            # save bbox to pickle
            with open(log_base+'-estimate_agents_bbox.pkl', 'wb') as f:
                pickle.dump(mov_track_bbox, f)
            with open(log_base+'-estimate_agents_bbox_label.pkl', 'wb') as f:
                pickle.dump(person_car_label, f)
            # np.save(log_base+'-estimate_agents_bbox.npy', mov_track_bbox)

        else:
            if os.path.exists(log_base+'-estimate_agents_traj.pkl'):
                os.system(f'rm -rf {log_base}-estimate_agents_traj.pkl')
            if os.path.exists(log_base+'-estimate_agents_bbox.pkl'):
                os.system(f'rm -rf {log_base}-estimate_agents_bbox.pkl')
            if os.path.exists(log_base+'-estimate_agents_bbox_label.pkl'):
                os.system(f'rm -rf {log_base}-estimate_agents_bbox_label.pkl')
