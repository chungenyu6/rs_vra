#!/bin/bash

python eval/POPE-n900/evaluate.py --subset random --sample 300 --model llava1.5 --wandb True
python eval/POPE-n900/evaluate.py --subset popular --sample 300 --model llava1.5 --wandb True
python eval/POPE-n900/evaluate.py --subset adversarial --sample 300 --model llava1.5 --wandb True


python eval/POPE-n900/evaluate.py --subset random --sample 300 --model agent --max_reflexion_iters 1 --wandb True
python eval/POPE-n900/evaluate.py --subset popular --sample 300 --model agent --max_reflexion_iters 1 --wandb True
python eval/POPE-n900/evaluate.py --subset adversarial --sample 300 --model agent --max_reflexion_iters 1 --wandb True