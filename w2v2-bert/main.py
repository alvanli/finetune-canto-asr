# https://huggingface.co/docs/datasets/audio_dataset
import os
import numpy as np
import datasets
from pathlib import Path

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from datasets import interleave_datasets, load_dataset, IterableDataset, IterableDatasetDict, Audio, load_from_disk
from transformers.trainer_pt_utils import IterableDatasetShard

from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

from transformers import TrainingArguments, TrainerCallback, Trainer

import evaluate

metric = evaluate.load("cer")

SAMPLING_RATE = 16_000

max_input_length = 30.0

def is_audio_in_length_range(length):
    return length < max_input_length


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


if __name__ == "__main__":
    ds_train_vect = load_from_disk("/exp/whisper_yue/whisper_data/combined_canto_train")
    ds_test_vect = load_from_disk("/exp/whisper_yue/whisper_data/cv_test")
    print("Loaded datasets")

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./tokenizer", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=SAMPLING_RATE, padding_value=0.0)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    print(len(ds_train_vect), len(ds_test_vect))
    ds_train_vect = ds_train_vect.filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )
    ds_test_vect = ds_test_vect.filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )
    print(len(ds_train_vect), len(ds_test_vect))


    ds_train_vect = ds_train_vect.shuffle(
        seed=0
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # "facebook/w2v-bert-2.0"
    model = Wav2Vec2BertForCTC.from_pretrained(
        "alvanlii/wav2vec2-BERT-cantonese",
        attention_dropout=0.1,
        hidden_dropout=0.0,
        feat_proj_dropout=0.1,
        mask_time_prob=0.0,
        layerdrop=0.1,
        ctc_zero_infinity=True,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    training_args = TrainingArguments(
        output_dir="canto_bertw2v2",
        # group_by_length=True,
        do_eval=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        evaluation_strategy="steps",
        num_train_epochs=1,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=600,
        # eval_steps=300,
        logging_steps=300,
        learning_rate=5e-5,
        warmup_steps=2000,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds_train_vect,
        eval_dataset=ds_test_vect,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training")
    trainer.train()
