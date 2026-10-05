#!/bin/sh

######## BEGIN TODO: experiment configuration ########
# Configuration only: these defaults are complete; 
# no implementation is required.
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./sample_data/dreambooth-cat"
export OUTPUT_DIR="./runs/dreambooth_cat"
INSTANCE_PROMPT="a photo of a sks cat"
MIXED_PRECISION="no"
RESOLUTION=512
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_STEPS=100
LEARNING_RATE="1e-4"
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
MAX_TRAIN_STEPS=500
VALIDATION_PROMPT="A photo of sks cat in a bucket"
VALIDATION_EPOCHS=50
CHECKPOINTS_TOTAL_LIMIT=2
SEED=0
######## END TODO: experiment configuration ########

accelerate launch --mixed_precision="$MIXED_PRECISION" train_dreambooth_lora.py \
  --mixed_precision="$MIXED_PRECISION" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --resolution="$RESOLUTION" \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  --checkpointing_steps="$CHECKPOINTING_STEPS" \
  --learning_rate="$LEARNING_RATE" \
  --lr_scheduler="$LR_SCHEDULER" \
  --lr_warmup_steps="$LR_WARMUP_STEPS" \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --validation_prompt="$VALIDATION_PROMPT" \
  --validation_epochs="$VALIDATION_EPOCHS" \
  --checkpoints_total_limit="$CHECKPOINTS_TOTAL_LIMIT" \
  --seed="$SEED"
