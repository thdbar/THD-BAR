

#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

OMP_NUM_THREADS=16 torchrun --master_port 29529 --nnodes=1 --nproc_per_node=8 train_instruction_BAR.py \
    --dataset_dir /data/home/username/dataset/test \
    --out_dir /data/home/username/code/BAR/output \
    --tokenizer_path ./checkpoints/MSVQ_wo_ema/ckpt.pt \
    --BAR_path checkpoints/..pt \
    --wandb_log \
    --wandb_project BAR \
    --entity='' \
    --wandb_runname BAR_ft \
    --wandb_api_key 1234 \
    --eeg_batch_size 40 \
    --learning_rate 6e-4 \
    --epochs 50 
