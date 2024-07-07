#!/usr/bin/env bash

python3 create_student_model.py \
  --teacher_checkpoint "Scrya/whisper-large-v2-cantonese" \
  --encoder_layers 32 \
  --decoder_layers 3 \
  --save_dir "./distil-large-v2-init"