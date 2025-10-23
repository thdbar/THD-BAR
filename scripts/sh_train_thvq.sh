#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

OMP_NUM_THREADS=16 torchrun --master_port 29501 --nnodes=1 --nproc_per_node=8 train_thvq.py \
    --dataset_dir /data/home/username/dataset/ \
    --out_dir /data/home/username/code/BAR/output \
    --wandb_log \
    --wandb_project BAR \
    --wandb_runname MSVQ \
    --wandb_api_key 1234




