# !/bin/bash

TRAIN_DATASET_NAME=$1 # for example: libero_90_no_noops
WANDB_NAME=$2
VAE_MODEL_PATH=train_vae/scripts/action_vqvae_config   # The vae model dir
OUTPUT_DIR=outputs/$TRAIN_DATASET_NAME/$WANDB_NAME    # The checkpoint saving dir
BATCH_SIZE=1024
STEPS=400000

torchrun --nproc_per_node 1 \
    train_vae/scripts/train_action_vqvae.py \
    --vqvae_config_path $VAE_MODEL_PATH \
    --model_dtype bf16 \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --seed 42 \
    --weight_decay 2e-4 \
    --clip_grad 1.0 \
    --lr 5e-5 \
    --print_freq 10000 \
    --save_ckpt_freq 100000 \
    --data_root_dir "/path/to/dataset" \
    --train_dataset_name $TRAIN_DATASET_NAME \
    --total_steps $STEPS --wandb_name $WANDB_NAME-$TRAIN_DATASET_NAME \
    --use_action_type_pe \
    --use_time_pe 
