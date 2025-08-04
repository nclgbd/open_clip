#!/bin/bash
# bash run_openclip.sh
# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# set the training args
train_data="/home/nicoleg/workspaces/dissertation/.data/Full_Data/train/combined_data.csv"
batch_size=24
lr=5e-6
warmup=100
wd=1.0
epochs=5

torchrun --nproc_per_node 4 -m open_clip_train.main \
    --batch-size ${batch_size} \
    --precision amp \
    --workers 24 \
    --report-to wandb \
    --save-frequency 1 \
    --val-frequency 1 \
    --dataset-type csv \
    --csv-separator="," \
    --train-data ${train_data} \
    --val-data /home/nicoleg/workspaces/dissertation/.data/Full_Data/validation/val_data.csv \
    --csv-img-key image_files \
    --csv-caption-key clip_prompts \
    --warmup ${warmup} \
    --lr ${lr} \
    --wd ${wd} \
    --epochs ${epochs} \
    --model 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
