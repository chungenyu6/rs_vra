#! /bin/bash

# Run: bash scripts/batch_eval_vrsbench_vqa.sh

cd /home/ISRAgent
########################################################################################
# Run one model on one question type
########################################################################################
# python eval/VRSBench_vqa-n1000/evaluate.py --model agent --max_reflexion_iters 2 --version rs_vra-rm1-vm1-aa1-ri2 --qtype reasoning --sample 1 --wandb True

# python eval/VRSBench_vqa-n1000/evaluate.py --model geochat --version v1 --qtype reasoning --sample 1 --wandb True

########################################################################################
# Run all models
# model: agent, geochat, llava15, gemma3, mistral31, gemini25-flash
## TODO: change model, version
########################################################################################
agent_version=rs_vra-rm1-vm4-aa1-ri3
lvlm_version=v1
sample=50
max_reflexion_iters=3
wandb=True
# object_quantity object_position object_direction object_size reasoning object_color object_existence object_category object_shape scene_type
for model in gemini25-flash; do
    for qtype in object_quantity object_position object_direction object_size reasoning object_color object_existence object_category object_shape scene_type; do
        echo "Evaluating $model on $qtype"
        if [ $model == "agent" ]; then
            python eval/VRSBench_vqa-n1000/evaluate.py --model $model --max_reflexion_iters $max_reflexion_iters --version $agent_version --qtype $qtype --sample $sample --wandb $wandb
        else
            python eval/VRSBench_vqa-n1000/evaluate.py --model $model --version $lvlm_version --qtype $qtype --sample $sample --wandb $wandb
        fi
    done
done