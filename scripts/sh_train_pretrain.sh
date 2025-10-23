#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

OMP_NUM_THREADS=16 torchrun --master_port 29520 --nnodes=1 --nproc_per_node=8 train_pretrain_BAR.py \
    --dataset_dir /data/home/username/dataset \
    --out_dir /data/home/username/code/BAR/output \
    --tokenizer_path ./checkpoints/MSVQ_wo_ema/ckpt.pt \
    --wandb_log \
    --wandb_project BAR \
    --entity '' \
    --wandb_runname BAR_pt \
    --wandb_api_key 1234 \
    --eeg_batch_size 4 \
    --learning_rate 1e-4 
