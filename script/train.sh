#!/bin/bash

# Model
MODEL_NAME="Qwen/Qwen3-0.6B" 

# Dataset
JSON_DATA_PATH="/workspace/llm_full_fine_tuning/data/example_dataset.json"

# Training configuration
GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=8
NUM_DEVICES=1
EPOCHS=100
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
OUTPUT_DIR=output/fft_$(date +%Y%m%d_%H%M%S)
DEEPSPEED_CONFIG=script/zero3.json

export PYTHONPATH=src:$PYTHONPATH

echo "💠 Model: $MODEL_NAME"
echo "💠 Data Path: $JSON_DATA_PATH"
echo "💠 Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "💠 Batch Per Device: $BATCH_PER_DEVICE"
echo "💠 Number of Devices: $NUM_DEVICES"
echo "💠 Grad Accum Steps: $GRAD_ACCUM_STEPS"
echo "💠 Epochs: $EPOCHS"
echo "💠 Output Dir: $OUTPUT_DIR"
echo "💠 Deepspeed Config: $DEEPSPEED_CONFIG"
echo ""
echo "🔥 Starting Training..."


deepspeed src/train/train_sft.py \
    --model_id $MODEL_NAME \
    --data_path $JSON_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --deepspeed $DEEPSPEED_CONFIG \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --use_liger True \
    --use_lora False \
    --use_dora False \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --remove_unused_columns False \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --tf32 True \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 5 \
    --dataloader_num_workers 4