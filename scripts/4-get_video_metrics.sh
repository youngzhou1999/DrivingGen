
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=./ckpt
export TORCH_HOME=./ckpt

model=wan2.2-14b
exp_id=default_prompt
split=ego_condition

metric=all

pwd
set -x
CUDA_VISIBLE_DEVICES='0' python drivinggen/z-sample_fvd.py \
    --root_path  ./cache/infer_results/${split} \
    --outdir     ./cache/eval_logs/${split} \
    --gt_path ./data/${split}.json \
    --track ${split} \
    --model_name "${model}" \
    --exp_id "${exp_id}" \
    --metric ${metric}

