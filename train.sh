#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=6 main.py --cfg ./config/train.yaml
