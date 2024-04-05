#!/usr/bin/env bash

python3 create_student_model.py \
  --teacher_checkpoint "alvanlii/whisper-small-cantonese" \
  --encoder_layers 12 \
  --decoder_layers 4 \
  --save_dir "./distil-small-init"