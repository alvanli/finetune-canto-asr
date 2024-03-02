#!/usr/bin/env bash

python3 create_student_model.py \
  --teacher_checkpoint "/exp/whisper_yue/finetune-whisper-canto/whisper_small/model_out/checkpoint-15000" \
  --encoder_layers 12 \
  --decoder_layers 3 \
  --save_dir "./distil-small-init"