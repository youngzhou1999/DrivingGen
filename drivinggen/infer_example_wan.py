import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
import json
import argparse
import os

import torch
import numpy as np
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

from PIL import Image
from einops import rearrange, repeat
import numpy as np
import imageio

def save_img_seq_to_video(out_path, img_seq, fps):
    # img_seq: np array
    writer = imageio.get_writer(out_path, fps=fps)
    for img in img_seq:
        writer.append_data(img)
    writer.close()

def perform_save_locally(save_path, samples, mode):
    assert mode in ["images", "grids", "videos"]
    if mode != 'videos':
        merged_path = os.path.join(save_path, mode)
        os.makedirs(merged_path, exist_ok=True)

    if mode == "images":
        frame_count = 0
        for sample in samples:
            sample = sample * 255.0
            image_save_path = os.path.join(merged_path, f"{frame_count:05}.png")
            Image.fromarray(sample.astype(np.uint8)).save(image_save_path)
            frame_count += 1
    elif mode == "videos":
        img_seq = samples * 255.0
        video_save_path = os.path.join(save_path, f"video.mp4")
        save_img_seq_to_video(video_save_path, img_seq.astype(np.uint8), 10)
    else:
        raise NotImplementedError

def deal_img(img_path):
    image = load_image(img_path)
    max_area = 576 * 1024
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image

if __name__ == "__main__":

    # 保存原函数
    _orig_solve = torch.linalg.solve

    def solve_with_cpu_fallback(A, B):
        try:
            return _orig_solve(A, B)
        except RuntimeError as e:
            msg = str(e).lower()
            if "cusolver" in msg or "cusolver_status_internal_error" in msg:
                # 小矩阵搬到 CPU 解再搬回 GPU
                X = _orig_solve(A.detach().float().cpu(), B.detach().float().cpu())
                return X.to(A.device, dtype=A.dtype)
            raise

    torch.linalg.solve = solve_with_cpu_fallback


    parser = argparse.ArgumentParser(
            description="Generate a image or video from a text prompt or image using Wan"
        )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from."
    )
    parser.add_argument(
        "--video_save_folder",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wan2.2-14b",
        help="ego_condition or open_domain"
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default="default_prompt",
        help="ego_condition or open_domain"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="ego_condition",
        help="ego_condition or open_domain"
    )

    args = parser.parse_args()

    model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    dtype = torch.bfloat16
    device = "cuda"

    pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)

    base_path = args.image[:-5] # remove .json

    if '.json' in args.image:
        with open(args.image, 'r') as f:
            data = json.load(f)
        prompts = []
        runs = []
        for d in data[:2]:  # debug here for fast test
            first_img = os.path.join(base_path, 'imgs', d+'.jpg')
            if not os.path.exists(first_img):
                first_img[-4:] = ".png"
                if not os.path.exists(first_img):
                    print(f'first image not found. {first_img}')

            prompt_path = os.path.join(base_path, 'caption', d+'.txt')

            with open(prompt_path) as f:
                prompt = f.read()
            prompts.append({
                'prompt': prompt,
                'visual_input': first_img
            })
            runs.append(d)  # data id

        for idx, prompt in enumerate(prompts):
            image_path = prompt['visual_input']
            prompt = prompt['prompt']

            image = deal_img(image_path)

            negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            generator = torch.Generator(device=device).manual_seed(0)
            output = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=576,
                width=1024,
                num_frames=101,
                guidance_scale=3.5,
                num_inference_steps=40,
                generator=generator,
            ).frames[0]

            # export_to_video(output, "5bit2v_output.mp4", fps=10)
            run = runs[idx]
            virtual_path = os.path.join(args.video_save_folder, args.split, run, args.model, args.exp_id)
            os.makedirs(virtual_path, exist_ok=True)
            # 1 + 100
            samples = output[:101]
            perform_save_locally(virtual_path, samples, "videos")
            perform_save_locally(virtual_path, samples, "images")

