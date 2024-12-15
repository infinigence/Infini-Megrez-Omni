DATA_PATH=$(pwd)/../data/train/records.jsonl
DATASET_PREFIX=$(pwd)/../data/train/
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=$(pwd)/test_finetune/$CURRENT_TIME
LOGGING_DIR=$(pwd)/test_finetune_log
MODEL_PATH=""

torchrun --nproc_per_node=2 finetune.py \
    --data_path $DATA_PATH \
    --dataset_prefix $DATASET_PREFIX \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --model_name_or_path $MODEL_PATH \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --deepspeed ds_config_zero2.json \
    --prediction_loss_only false \
    --bf16 true \
    --fp16 false \
    --do_train \
    --tune_vision_encoder true \
    --tune_vision_proj true \
    --tune_llm true \
    --tune_audio_encoder false \
    --tune_audio_proj true \
    --model_max_length 2048 \
    --max_slice_nums 9 \
    --scale_resolution 448 \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --save_total_limit 100 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.98 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1
