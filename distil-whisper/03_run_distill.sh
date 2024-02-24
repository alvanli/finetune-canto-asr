python3 run_distillation.py \
  --model_name_or_path "./distil-small-init" \
  --teacher_model_name_or_path "alvanlii/whisper-small-cantonese" \
  --eval_steps 500 \
  --save_steps 500 \
  --warmup_steps 2000 \
  --learning_rate 0.00005 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 50 \
  --save_total_limit 3 \
  --max_steps 20000 \
  --cer_threshold 15 \
  --per_device_train_batch_size 48 \
  --per_device_eval_batch_size 24 \
  --dataloader_num_workers 16 \
  --preprocessing_num_workers 16 \
  --ddp_timeout 7200 \
  --dtype "bfloat16" \
  --output_dir "./distilled_boi_01" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --language "zh" \
  --timestamp_probability 0.0 \
  --use_pseudo_labels \
  --gradient_accumulation_steps 2  \
  --weight_decay 0.9