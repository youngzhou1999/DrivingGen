
## video pattern
## /mnt/cache/zhouyang/dg-bench/infer_logs_align_0822/beijing+DAY+WEATHER_NORMAL+static_scenes+crossing_turn_left+134336_s20-468_1700901918.0_1700901948.0/cogvideo/caption-frames_101-conds_1

# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=./ckpt
export TORCH_HOME=./ckpt


model=wan2.2-14b
exp_id=default_prompt
split=ego_condition

all_id=1
gpu_id=0

CUDA_VISIBLE_DEVICES=0 \
        python drivinggen/func/extract_traj_agent_unidepth.py \
          --root_path  cache/infer_results/${split} \
          --outdir     cache/eval_logs/${split} \
          --gt_meta_path data/${split}.json \
          --model_name      "${model}" \
          --exp_id     "${exp_id}" \
          --local_id   "${gpu_id}" \
          --all_id     "${all_id}"
