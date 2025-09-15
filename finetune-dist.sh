export CUDA_VISIBLE_DEVICES=2,3
uv run --no-sync torchrun --nproc_per_node 2 -m open_clip_train.main \
    --save-frequency 3 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="train_data.csv"  \
    --val-data="validate_data.csv"  \
    --csv-img-key image \
    --csv-caption-key label \
    --warmup 1000 \
    --batch-size=32 \
    --lr=1e-6 \
    --wd=2.0 \
    --epochs=30 \
    --workers=16 \
    --csv-separator=, \
    --model 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'