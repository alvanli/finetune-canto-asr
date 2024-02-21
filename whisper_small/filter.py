# https://huggingface.co/docs/datasets/audio_dataset
import os
import datasets
from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path("/exp/hf_ds")

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from datasets import interleave_datasets, load_dataset, IterableDataset, IterableDatasetDict, Audio, load_from_disk
from transformers.trainer_pt_utils import IterableDatasetShard

from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments
from transformers import TrainerCallback, Seq2SeqTrainer

import evaluate

metric = evaluate.load("cer")

do_lower_case = False
do_remove_punctuation = False

max_input_length = 29.5
torch.set_default_device("cuda")

SAVE_DIR = "/exp/whisper_yue/whisper_data"
PRETRAINED_MODEL = "alvanlii/whisper-small-cantonese"
model = WhisperForConditionalGeneration.from_pretrained(PRETRAINED_MODEL)
processor = WhisperProcessor.from_pretrained(PRETRAINED_MODEL, task="transcribe")

def is_audio_in_length_range(length):
    return length < max_input_length

def is_audio_right_size(input_features):
    return len(input_features)==80 and len(input_features[0])==3000

def is_audio_ok_for_model(input_features, labels):
    input_features_new = [{"input_features": torch.Tensor(input_features)}]
    labels_new = [{"input_ids": torch.Tensor(labels).long()}]
    batch = processor.feature_extractor.pad(input_features_new, return_tensors="pt")
    labels_batch = processor.tokenizer.pad(labels_new, return_tensors="pt")

    try:
        model(**batch, labels=labels_batch["input_ids"])
        return True
    except:
        return False

if __name__ == "__main__":
    ds_train_vect = load_from_disk(f"{SAVE_DIR}/combined_canto_train")

    print(len(ds_train_vect))
    ds_train_vect = ds_train_vect.filter(
        is_audio_in_length_range,
        input_columns=["input_length"]
    )
    print(len(ds_train_vect))
    ds_train_vect = ds_train_vect.filter(
        is_audio_ok_for_model,
        input_columns=["input_features", "labels"]
    )
    # ds_train_vect = ds_train_vect.filter(
    #     is_audio_right_size,
    #     input_columns=["input_features"],
    # )
    print(len(ds_train_vect))

    ds_train_vect.save_to_disk(f"{SAVE_DIR}/combined_canto_train_filtered")
