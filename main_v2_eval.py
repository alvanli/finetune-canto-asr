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

    saved_ds_dir = "/data2/processed_canto"
    processed_dir = saved_ds_dir+"/aug_combined_train_filtered"
    WHISPER_MODEL = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, task="transcribe")
    
    print("Loaded processor")

    ds_test_vect = load_from_disk("/data2/processed_common_11_hk/test")

    print("Loaded datasets")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


    from peft import PeftModel, PeftConfig
    from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer

    # peft_model_id = "alvanlii/whisper-largev2-cantonese-peft-lora"
    peft_model_id = "./model_out_largev2_further/checkpoint-20000/adapter_model"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)

    # model.get_base_model().save_pretrained(f"./{peft_model_id}/onnx")

    # from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

    # onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
    #     f"./{peft_model_id}/onnx", from_transformers=True, provider="CUDAExecutionProvider"
    # )

    print("Starting Evaluation")
    
    import gc
    import numpy as np
    from torch.utils.data import DataLoader

    metric = evaluate.load("cer")

    eval_dataloader = DataLoader(ds_test_vect, batch_size=32, collate_fn=data_collator)

    model.eval()
    for step, batch in enumerate(eval_dataloader):
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
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                decoded_preds = [normalizer(pred) for pred in decoded_preds]
                decoded_labels = [normalizer(label) for label in decoded_labels]

                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )

        del generated_tokens, labels, batch
        gc.collect()
    
    print("Original model: ", metric.compute())
    del metric, eval_dataloader




    # metric = evaluate.load("cer")

    # eval_dataloader = DataLoader(ds_test_vect, batch_size=20, collate_fn=data_collator)

    # # onnx_model.eval()
    # for step, batch in enumerate(eval_dataloader):
    #     with torch.cuda.amp.autocast():
    #         with torch.no_grad():
    #             generated_tokens = (
    #                 onnx_model.generate(
    #                     input_features=batch["input_features"].to("cuda"),
    #                     decoder_input_ids=batch["labels"][:, :4].to("cuda"),
    #                     max_new_tokens=255,
    #                 )
    #                 .cpu()
    #                 .numpy()
    #             )
    #             labels = batch["labels"].cpu().numpy()
    #             labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                
    #             decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    #             decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
    #             decoded_preds = [normalizer(pred) for pred in decoded_preds]
    #             decoded_labels = [normalizer(label) for label in decoded_labels]

    #             metric.add_batch(
    #                 predictions=decoded_preds,
    #                 references=decoded_labels,
    #             )

    #     del generated_tokens, labels, batch
    #     gc.collect()
    
    # print("ONNX model: ", metric.compute())