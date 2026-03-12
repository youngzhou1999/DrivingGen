import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
# sys.path.append("./sam2")
sys.path.append("/shared_disk/users/yang.zhou/iclr_open_source/samurai/sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def samurai_main(
        model_path='/shared_disk/users/yang.zhou/iclr_open_source/DrivingGen/ckpt/sam2.1_hiera_large.pt', 
        video_path=None,
        txt_path=None,
        save_to_video=False,
        seg_out_dir='./',
    ):
    model_cfg = determine_model_cfg(model_path)
    predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
    # frames_or_path = prepare_frames_or_path(video_path)
    frames_or_path = video_path
    # prompts = load_txt(txt_path)
    prompts = txt_path

    frame_rate = 30
    frames = frames_or_path
    loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
    height, width = loaded_frames[0].shape[:2]
    if height == 2400:
        height = 600
        # if osp.isdir(video_path):
        #     frames = sorted([osp.join(video_path, f) for f in os.listdir(video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
        #     loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
        #     height, width = loaded_frames[0].shape[:2]
        # else:
        #     cap = cv2.VideoCapture(video_path)
        #     frame_rate = cap.get(cv2.CAP_PROP_FPS)
        #     loaded_frames = []
        #     while True:
        #         ret, frame = cap.read()
        #         if not ret:
        #             break
        #         loaded_frames.append(frame)
        #     cap.release()
        #     height, width = loaded_frames[0].shape[:2]

        #     if len(loaded_frames) == 0:
        #         raise ValueError("No frames were loaded from the video.")

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(video_output_path, fourcc, frame_rate, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        # bbox, track_label = prompts[0]
        bbox = np.array(prompts, dtype=np.float32)
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        mask_frames = []
        bbox_frames = []
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            # how about no masks
            empty_mask = np.zeros((height, width), bool)
            empty_bbox = None
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                empty_mask = empty_mask | mask
                empty_mask = empty_mask.astype(np.uint8)
                empty_mask[empty_mask > 0] = 255
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    empty_bbox = [x_min, y_min, x_max, y_max]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask
            
            mask_frames.append(empty_mask)  # suitable for one object
            if empty_bbox is not None:
                bbox_frames.append((frame_idx, empty_bbox))

            if save_to_video:
                img = loaded_frames[frame_idx]
                if img.shape[0] == 2400:
                    img = img[:600, :, :]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

                # save as tracking
                cv2.imwrite(os.path.join(seg_out_dir, f'{frame_idx:05}.png'), img)

        #         out.write(img)

        # if save_to_video:
        #     out.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

    return mask_frames, bbox_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    args = parser.parse_args()
    samurai_main(args)