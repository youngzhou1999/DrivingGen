pwd
set -x

export WANDB_API_KEY=ec6f304c3028e945e9b6bf89e264910f8bbbcdc3
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/cache/zhouyang/ckpts/huggingface
export TORCH_HOME=/mnt/cache/zhouyang/ckpts/huggingface

CUDA_VISIBLE_DEVICES='3' python scripts/inference.py \
    --prompt prompts/question.yaml \
    --question 'Does the car in the green bounding box disappear abnormally, based on its visual continuity and its interactions with surrounding vehicles and the environment?' \
    --reasoning \
    --images \
    /mnt/cache/zhouyang/dg-bench/eval_suites/outputs/eval_logs_align_mini_0822/glm_input/vista/traj-frames_25-conds_1-rounds_5/beijing+DAY+WEATHER_SNOW+open-world_dynamics+traffic_light_red_to_green+095342_s20-271_1702871699.0_1702871725.0/4/00000.png \
    /mnt/cache/zhouyang/dg-bench/eval_suites/outputs/eval_logs_align_mini_0822/glm_input/vista/traj-frames_25-conds_1-rounds_5/beijing+DAY+WEATHER_SNOW+open-world_dynamics+traffic_light_red_to_green+095342_s20-271_1702871699.0_1702871725.0/4/00001.png \
    /mnt/cache/zhouyang/dg-bench/eval_suites/outputs/eval_logs_align_mini_0822/glm_input/vista/traj-frames_25-conds_1-rounds_5/beijing+DAY+WEATHER_SNOW+open-world_dynamics+traffic_light_red_to_green+095342_s20-271_1702871699.0_1702871725.0/4/00074.png \
    /mnt/cache/zhouyang/dg-bench/eval_suites/outputs/eval_logs_align_mini_0822/glm_input/vista/traj-frames_25-conds_1-rounds_5/beijing+DAY+WEATHER_SNOW+open-world_dynamics+traffic_light_red_to_green+095342_s20-271_1702871699.0_1702871725.0/4/00078.png \
    /mnt/cache/zhouyang/dg-bench/eval_suites/outputs/eval_logs_align_mini_0822/glm_input/vista/traj-frames_25-conds_1-rounds_5/beijing+DAY+WEATHER_SNOW+open-world_dynamics+traffic_light_red_to_green+095342_s20-271_1702871699.0_1702871725.0/4/00079.png \
    /mnt/cache/zhouyang/dg-bench/eval_suites/outputs/eval_logs_align_mini_0822/glm_input/vista/traj-frames_25-conds_1-rounds_5/beijing+DAY+WEATHER_SNOW+open-world_dynamics+traffic_light_red_to_green+095342_s20-271_1702871699.0_1702871725.0/4/00080.png \
    -v

# CUDA_VISIBLE_DEVICES='3' python scripts/inference_sample.py
