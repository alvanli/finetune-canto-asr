# https://huggingface.co/docs/datasets/audio_dataset
import os
os.environ['LD_LIBRARY_PATH']='/usr/lib/x86_64-linux-gnu/:/opt/conda/lib/'
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
    peft_model_id = "alvanlii/whisper-largev2-cantonese-peft-lora"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    # model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL, load_in_8bit=True, device_map="auto")
    # print("Loaded model")

    # from peft import prepare_model_for_int8_training
    # model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")

    # from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

    # config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

    # model = get_peft_model(model, config)
    model.print_trainable_parameters()

    print("Loaded PEFT LORA Model")

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False


    saved_ds_dir = "/data2/processed_canto"
    # processed_dir = "/data2/processed_common_11_hk/combined_train_filtered"
    processed_dir = saved_ds_dir+"/aug_combined_train_filtered"
    WHISPER_MODEL = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, task="transcribe")
    
    print("Loaded processor")

    if not os.path.exists(processed_dir):
        ds_train_vect = load_from_disk("/data2/aug_combined_canto")
        ds_test_vect = load_from_disk("/data2/processed_common_11_hk/test")

        ds_train_vect = ds_train_vect.shuffle(
            seed=0
        )
        print(f"Before filtering {len(ds_train_vect)}")
        ds_train_vect = ds_train_vect.filter(
            is_audio_in_length_range,
            input_columns=["input_length"],
        )
        print(f"After filtering {len(ds_train_vect)}")
        ds_train_vect.save_to_disk(processed_dir)
    else:
        ds_train_vect = load_from_disk(processed_dir)
        ds_test_vect = load_from_disk("/data2/processed_common_11_hk/test")

    print("Loaded datasets")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        if do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]

        cer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}


    output_dir="./model_out_largev2_further"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=60,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=5e-4,
        warmup_steps=500,
        max_steps=10000,
        gradient_checkpointing=True,
        num_train_epochs=5,
        evaluation_strategy="steps",
        per_device_eval_batch_size=32,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        fp16=True,
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
        # compute_metrics=compute_metrics,
        tokenizer=processor,
        callbacks=[ShuffleCallback()],
    )


    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    trainer.train()


    print("Starting Evaluation")
    
    import gc
    import numpy as np
    from torch.utils.data import DataLoader

    eval_dataloader = DataLoader(ds_test_vect, batch_size=10, collate_fn=data_collator)

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        del generated_tokens, labels, batch
        gc.collect()
    
    print(metric.compute())