# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=./ckpt
export TORCH_HOME=./ckpt

video_path=data/ego_condition.json
out_dir=cache/infer_results
split=ego_condition
model=wan2.2-14b
exp_id=default_prompt

CUDA_VISIBLE_DEVICES='0' python drivinggen/infer_example_wan.py \
    --image $video_path \
    --video_save_folder $out_dir \
    --split $split \
    --model $model \
    --exp_id $exp_id
