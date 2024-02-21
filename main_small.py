# https://huggingface.co/docs/datasets/audio_dataset
import os
import datasets
from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path("/exp/hf_ds")

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
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

normalizer = BasicTextNormalizer()
max_input_length = 30.0
max_label_length = 448

def is_audio_in_length_range(length):
    return length < max_input_length

def is_label_in_length_range(labels):
    return len(labels) < max_label_length

def is_audio_right_size(input_features):
    return len(input_features)==80 and len(input_features[0])==3000

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

class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)


SAVE_DIR = "/exp/whisper_yue/whisper_data"
PRETRAINED_MODEL = "alvanlii/whisper-small-cantonese"
if __name__ == "__main__":
    processor = WhisperProcessor.from_pretrained(PRETRAINED_MODEL, task="transcribe")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    model = WhisperForConditionalGeneration.from_pretrained(PRETRAINED_MODEL)

    print("Loaded model")

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    ds_train_vect = load_from_disk(f"{SAVE_DIR}/combined_canto_train_filtered")
    ds_test_vect = load_from_disk(f"{SAVE_DIR}/cv_test")

    print(len(ds_train_vect))
    # ds_train_vect = ds_train_vect.filter(
    #     is_audio_in_length_range,
    #     input_columns=["input_length"],
    # )
    ds_train_vect = ds_train_vect.filter(
        is_label_in_length_range,
        input_columns=["labels"],
    )
    # ds_train_vect = ds_train_vect.filter(
    #     is_audio_right_size,
    #     input_columns=["input_features"],
    # )
    print(len(ds_train_vect))

    print("Loaded datasets")


    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    output_dir="./model_out"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=25,
        gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
        learning_rate=5e-5,
        warmup_steps=500,
        max_steps=15000,
        gradient_checkpointing=True,
        num_train_epochs=5,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=False,
        save_total_limit=5
    )

    print("Starting Trainer...")

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds_train_vect,
        eval_dataset=ds_test_vect,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        callbacks=[ShuffleCallback()],
    )


    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    trainer.train()
