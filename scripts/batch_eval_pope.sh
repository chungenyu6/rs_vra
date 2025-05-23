#!/bin/bash

cd /home/ISRAgent
# python eval/POPE-n900/evaluate.py --subset random --sample 1 --model llava1.5 --version v1 --wandb False

python eval/POPE-n900/evaluate.py --subset random --sample 300 --model agent --max_reflexion_iters 1 --version vra-rm1.1-vm1-aa1-ri1 --wandb True
python eval/POPE-n900/evaluate.py --subset popular --sample 300 --model agent --max_reflexion_iters 1 --version vra-rm1.1-vm1-aa1-ri1 --wandb True
python eval/POPE-n900/evaluate.py --subset adversarial --sample 300 --model agent --max_reflexion_iters 1 --version vra-rm1.1-vm1-aa1-ri1 --wandb True