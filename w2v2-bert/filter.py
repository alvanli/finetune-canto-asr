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

from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

from transformers import TrainerCallback, Seq2SeqTrainer

import evaluate

metric = evaluate.load("cer")

do_lower_case = False
do_remove_punctuation = False

max_input_length = 30.0
torch.set_default_device("cuda")

SAMPLING_RATE = 16_000
SAVE_DIR = "/exp/whisper_yue/w2v2_data"
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./tokenizer", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=SAMPLING_RATE, padding_value=0.0)
processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


PRETRAINED_PATH = "/exp/whisper_yue/finetune-whisper-canto/w2v2-bert/canto_bertw2v2/checkpoint-600"
model = Wav2Vec2BertForCTC.from_pretrained(
    PRETRAINED_PATH,
    attention_dropout=0.1,
    hidden_dropout=0.0,
    feat_proj_dropout=0.1,
    mask_time_prob=0.0,
    layerdrop=0.1,
    ctc_zero_infinity=True,
    ctc_loss_reduction="mean",
    add_adapter=True,
    pad_token_id=tokenizer.pad_token_id,
    vocab_size=len(tokenizer),
)
model = model

def is_audio_in_length_range(length):
    return length < max_input_length

def is_audio_ok_for_model(input_features, labels):
    input_features_new = [{"input_features": torch.Tensor(input_features)}]
    labels_new = [{"input_ids": torch.Tensor(labels).long()}]
    batch = processor.feature_extractor.pad(input_features_new, return_tensors="pt")
    labels_batch = processor.tokenizer.pad(labels_new, return_tensors="pt")
    # model(**batch, labels=labels_batch["input_ids"])
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
    print(len(ds_train_vect))

    ds_train_vect.save_to_disk(f"{SAVE_DIR}/combined_canto_train_filtered")
