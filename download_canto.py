# https://huggingface.co/docs/datasets/audio_dataset
import os
import datasets
from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path("/data/hf_cache")

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from datasets import interleave_datasets, load_dataset, IterableDataset, IterableDatasetDict, Audio, load_from_disk
from transformers.trainer_pt_utils import IterableDatasetShard

from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments
from transformers import TrainerCallback, Seq2SeqTrainer

import evaluate

metric = evaluate.load("wer")

# canto_tok = AutoTokenizer.from_pretrained("./tokenizer/tokenizer-canto")

do_lower_case = False
do_remove_punctuation = False

normalizer = BasicTextNormalizer()
max_input_length = 30.0

def is_audio_in_length_range(length):
    return length < max_input_length

# evaluate with the 'normalised' WER
do_normalize_eval = True


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


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
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    transcription = batch["sentence"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)


if __name__ == "__main__":

    saved_ds_dir = "/data/processed_common_11_hk"
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")

    ds_train = load_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="train+validation", use_auth_token=True)  # set split="train+validation" for low-resource
    ds_test = load_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="test", use_auth_token=True)


    print(f"These are the ds features {ds_train.features}")

    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
    ds_train_vect = ds_train.map(prepare_dataset, remove_columns=list(ds_train.features.keys())).with_format("torch")
    ds_train_vect.save_to_disk(saved_ds_dir+"/train")

    ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16000))
    ds_test_vect = ds_test.map(prepare_dataset, remove_columns=list(ds_test.features.keys())).with_format("torch")
    ds_test_vect.save_to_disk(saved_ds_dir+"/test")


