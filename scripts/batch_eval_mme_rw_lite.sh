#! /bin/bash

# Run: bash scripts/batch_eval_vrsbench_vqa.sh

cd /home/ISRAgent
########################################################################################
# Run one model on one question type
########################################################################################
# python eval/MME-RealWorld-Lite/evaluate.py --model agent --max_reflexion_iters 2 --version rs_vra-rm1-vm1-aa1-ri2 --l2category reasoning --sample 1 --wandb True

# python eval/MME-RealWorld-Lite/evaluate.py --model geochat --version v1 --l2category reasoning --sample 1 --wandb True

########################################################################################
# Run all models
# model: agent, geochat, llava15, gemma3, mistral31, gemini25-flash
## TODO: change model, version
########################################################################################
agent_version=rs_vra-rm1-vm1-aa3-ri3
lvlm_version=v1
sample=50
max_reflexion_iters=3
wandb=True

# The folllowings does not include the l2-category within Perception/Diagram and Table, Perception/OCR with Complex Context, Reasoning/Diagram and Table, Reasoning/OCR with Complex Context

for model in agent; do
    for l2category in attribute_motion_multipedestrians attribute_motion_multivehicles attribute_motion_pedestrian attribute_motion_vehicle attribute_visual_trafficsignal object_count objects_identify person_attribute_color person_counting vehicle_attribute_color vehicle_attribute_orientation vehicle_counting vehicle_location color count position attention_trafficsignal prediction_intention_ego prediction_intention_pedestrian prediction_intention_vehicle relation_interaction_ego2pedestrian relation_interaction_ego2trafficsignal relation_interaction_ego2vehicle relation_interaction_other2other calculate intention property; do
        echo "Evaluating $model on $l2category"
        if [ $model == "agent" ]; then
            python eval/MME-RealWorld-Lite/evaluate.py --model $model --max_reflexion_iters $max_reflexion_iters --version $agent_version --l2category $l2category --sample $sample --wandb $wandb
        else
            python eval/MME-RealWorld-Lite/evaluate.py --model $model --version $lvlm_version --l2category $l2category --sample $sample --wandb $wandb
        fi
    done
done

# The followings has different sample numbers: person_attribute_orientation
sample_2=19

for model in agent; do
    for l2category in person_attribute_orientation; do
        echo "Evaluating $model on $l2category"
        if [ $model == "agent" ]; then
            python eval/MME-RealWorld-Lite/evaluate.py --model $model --max_reflexion_iters $max_reflexion_iters --version $agent_version --l2category $l2category --sample $sample_2 --wandb $wandb
        else
            python eval/MME-RealWorld-Lite/evaluate.py --model $model --version $lvlm_version --l2category $l2category --sample $sample_2 --wandb $wandb
        fi
    done
done