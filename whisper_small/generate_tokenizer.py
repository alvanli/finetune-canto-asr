import string
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from datasets import interleave_datasets, load_dataset, IterableDatasetDict, Audio
from transformers import WhisperProcessor


do_lower_case = False
do_remove_punctuation = False

punctuation_to_remove = string.punctuation.replace("'", "")  # don't remove apostrophes
punctuation_to_remove_regex = f"[{''.join(punctuation_to_remove)}]"

if do_remove_punctuation:
    print("Removing punctuation: ", punctuation_to_remove)

max_input_length = 30.0

def is_audio_in_length_range(length):
    return length < max_input_length

# evaluate with the 'normalised' WER
do_normalize_eval = True

def load_streaming_dataset(dataset_name, dataset_config_name, split, **kwargs):
    if "+" in split:
        # load multiple splits separated by the `+` symbol *with* streaming mode
        dataset_splits = [load_dataset(dataset_name, dataset_config_name, split=split_name, streaming=True, **kwargs) for split_name in split.split("+")]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(dataset_name, dataset_config_name, split=split, streaming=True, **kwargs)
        return dataset


def prepare_dataset(batch):
    # optional pre-processing steps
    batch["transcription"] = batch["sentence"]
    return batch


if __name__ == "__main__":

    raw_datasets = IterableDatasetDict()

    raw_datasets["train"] = load_streaming_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="train", use_auth_token=True)  # set split="train+validation" for low-resource
    raw_datasets["test"] = load_streaming_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="test", use_auth_token=True)

    print(f"These are the ds features {raw_datasets['train'].features}")

    vectorized_datasets = raw_datasets.map(prepare_dataset, remove_columns=list(next(iter(raw_datasets.values())).features))

    with open("/exp/whisper_yue/finetune-whisper-canto/just_text/all_texts.txt", "a") as f:
        for row in vectorized_datasets["train"]:
            f.write(row['transcription'])
            
        print("Done train")
        
        for row in vectorized_datasets["test"]:
            f.write(row['transcription'])

    