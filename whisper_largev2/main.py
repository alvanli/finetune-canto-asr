# https://huggingface.co/docs/datasets/audio_dataset
import os, glob
os.environ['LD_LIBRARY_PATH']='/usr/lib/x86_64-linux-gnu/:/opt/conda/lib/'
import datasets
from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path("/exp/hf_ds")
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

from datasets import load_from_disk, concatenate_datasets
from transformers.trainer_pt_utils import IterableDatasetShard

from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments
from transformers import TrainerCallback, Seq2SeqTrainer
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, PeftConfig
import evaluate

WHISPER_MODEL = "openai/whisper-large-v2"
LANGUAGE = "zh"
TASK = "transcribe"
TRAIN_FROM_SCRATCH = False
SAVE_DIR = "/exp/whisper_yue/whisper_data"


metric = evaluate.load("cer")

do_lower_case = False
do_remove_punctuation = False

max_input_length = 30.0
max_label_length = 448

def is_audio_in_length_range(length):
    return length < max_input_length

def is_label_in_length_range(labels):
    return len(labels) < max_label_length

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


if __name__ == "__main__":
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, language=LANGUAGE, task=TASK) 
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL, load_in_8bit=False)

    if TRAIN_FROM_SCRATCH: 
        output_dir="./model_out_01"

        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        print("Loaded PEFT LORA Model")

        config = LoraConfig(use_dora=True, r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        model.config.use_cache = False

        # as Whisper model uses Conv layer in encoder, checkpointing disables grad computation
        # to avoid this, make the inputs trainable
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

        ds_train_vect = load_from_disk(f"{SAVE_DIR}/combined_canto_train")
    else:
        output_dir="./model_out_02"
        peft_model_id = "/exp/whisper_yue/finetune-whisper-canto/whisper_largev2/model_out_01/checkpoint-10000"   

        model = PeftModel.from_pretrained(model, peft_model_id, is_trainable=True)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

        ds_canto_asr = [load_from_disk(dir) for dir in glob.glob(f"{SAVE_DIR}/canto_asr*")]
        ds_canto_map = [load_from_disk(dir) for dir in glob.glob(f"{SAVE_DIR}/canto_map*")]
        cv_trains = [load_from_disk(f"{SAVE_DIR}/cv_train"), load_from_disk(f"{SAVE_DIR}/cv_train_2")]
        ds_train_vect = concatenate_datasets(cv_trains + ds_canto_asr + ds_canto_map)


    print(f"Train Len: {ds_train_vect}")
    ds_train_vect = ds_train_vect.filter(
        is_audio_in_length_range,
        input_columns=["input_length"]
    )
    ds_train_vect = ds_train_vect.filter(
        is_label_in_length_range,
        input_columns=["labels"],
    )
    print(f"Train Len: {ds_train_vect}")
    ds_test_vect = load_from_disk(f"{SAVE_DIR}/cv_test")
    print("Loaded datasets")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=14,
        gradient_accumulation_steps=8,  # increase by 2x for every 2x decrease in batch size
        learning_rate=5e-5,
        warmup_steps=1000 if TRAIN_FROM_SCRATCH else 0,
        gradient_checkpointing=True,
        num_train_epochs=3 if TRAIN_FROM_SCRATCH else 5,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=10,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500 if TRAIN_FROM_SCRATCH else 200,
        eval_steps=500 if TRAIN_FROM_SCRATCH else 200,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=3,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )

    print("Starting Trainer...")

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds_train_vect,
        eval_dataset=ds_test_vect,
        data_collator=data_collator,
        tokenizer=processor
    )

    trainer.train()
  