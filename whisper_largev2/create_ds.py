import glob, os, re, json
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets, Audio
from transformers import WhisperProcessor, WhisperTokenizerFast

from bs4 import BeautifulSoup as bs
import librosa  
import soundfile as sf

BASE_DIR = "/exp/whisper_yue/"
SAVE_DIR = "/exp/whisper_yue/whisper_data"
SAMPLING_RATE = 16_000


def load_canto_map(processor, tokenizer):
    input_dir = BASE_DIR + "/CantoMap/processed"
    with open(input_dir+"/annots.json", "r", encoding="utf-8") as f:
        lines = f.read()
        line_dict = json.loads(lines)
    info_dict = {
        "input_features": [],
        "input_length": [],
        "labels": []
    }
    count_idx = 1
    for idx, key in enumerate(tqdm(list(line_dict.keys())[:])):
        curr_val = line_dict[key]
        if len(curr_val.strip()) > 1:
            arr, rate = librosa.load(f"{input_dir}/{key}.wav", sr=SAMPLING_RATE)
            try:
                feature = processor.feature_extractor(arr, sampling_rate=rate).input_features[0]
                clean_sentence = curr_val
                file_tokens = tokenizer(key, add_special_tokens=False).input_ids
                input_ids = processor(text=clean_sentence).input_ids

                info_dict["input_features"].append(feature),
                info_dict["input_length"].append(len(arr) / rate)
                info_dict["labels"].append(input_ids)
            except ValueError:
                print("Bad error")
        if idx % 5000 == 0 and idx != 0:
            ds = Dataset.from_dict(mapping=info_dict)
            ds.save_to_disk(f"{SAVE_DIR}/canto_map_{count_idx}")
            count_idx += 1
            info_dict = {
                "input_features": [],
                "input_length": [],
                "labels": []
            }   
    ds = Dataset.from_dict(mapping=info_dict)
    ds.save_to_disk(f"{SAVE_DIR}/canto_map_end")
    return


def load_canto_asr(processor, tokenizer):
    audio_base_path = BASE_DIR + "/cantonese-asr/dataset/data/audio"
    transcript_base_path = BASE_DIR + "/cantonese-asr/dataset/data/transcription"
    all_audio_files = glob.glob(audio_base_path+"/*.wav")
    
    info_dict = {
        "input_features": [],
        "input_length": [],
        "labels": []
    }
    count_idx = 1
    for idx, audio_file in enumerate(tqdm(all_audio_files)):  
        transcript_file = audio_file.replace("audio", "transcription").replace(".wav", ".txt")
        with open(transcript_file, "r", encoding="utf-8") as f:
            content = f.read()
        arr, rate = librosa.load(audio_file, sr=SAMPLING_RATE)
        input_length = len(arr) / rate
        if len(content.strip()) > 1:
            feature = processor.feature_extractor(arr, sampling_rate=rate).input_features[0]
            file_tokens = tokenizer(os.path.basename(audio_file), add_special_tokens=False).input_ids
            clean_sentence = content
            input_ids = processor(text=clean_sentence).input_ids

            info_dict["input_features"].append(feature)
            info_dict["input_length"].append(input_length)
            info_dict["labels"].append(input_ids)

        if idx % 5000 == 0 and idx != 0:
            ds = Dataset.from_dict(mapping=info_dict)
            ds.save_to_disk(f"{SAVE_DIR}/canto_asr_{count_idx}")
            count_idx += 1
            info_dict = {
                "input_features": [],
                "input_length": [],
                "labels": []
            }    

    ds = Dataset.from_dict(mapping=info_dict)
    ds.save_to_disk(f"{SAVE_DIR}/canto_asr_end")
    return


def merge_map_asr():
    ds_canto_asr = [load_from_disk(f"/exp/whisper_yue/whisper_data/canto_asr_{idx}") for idx in range(1,18)]
    ds_canto_map = [load_from_disk(f"/exp/whisper_yue/whisper_data/canto_map_{idx}") for idx in range(1,9)]
    ds_train_vect = load_from_disk("/exp/whisper_yue/whisper_data/cv_train")
    ds_train_vect_2 = load_from_disk("/exp/whisper_yue/whisper_data/cv_train_2")
    big_audio_ds = concatenate_datasets(ds_canto_asr + ds_canto_map + [ds_train_vect, ds_train_vect_2])
    big_audio_ds.save_to_disk(f"{SAVE_DIR}/combined_canto_train")


def load_cv(processor, tokenizer):
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    ds_train = load_dataset("mozilla-foundation/common_voice_16_0", "yue", split="train", use_auth_token=True)  
    ds_train_2 = load_dataset("mozilla-foundation/common_voice_16_0", "zh-HK", split="train", use_auth_token=True)  
    ds_test = load_dataset("mozilla-foundation/common_voice_16_0", "yue", split="test", use_auth_token=True)

    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16_000))
    ds_train_2 = ds_train_2.cast_column("audio", Audio(sampling_rate=16_000))
    ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16_000))

    ds_train = ds_train.map(prepare_dataset, remove_columns=ds_train.column_names)
    ds_train_2 = ds_train_2.map(prepare_dataset, remove_columns=ds_train_2.column_names)
    ds_test = ds_test.map(prepare_dataset, remove_columns=ds_test.column_names)

    ds_train.save_to_disk(f"{SAVE_DIR}/cv_train")
    ds_train_2.save_to_disk(f"{SAVE_DIR}/cv_train_2")
    ds_test.save_to_disk(f"{SAVE_DIR}/cv_test")
    return


if __name__ == "__main__":
    model_path = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_path, task="transcribe", language="yue")
    tokenizer = WhisperTokenizerFast.from_pretrained(model_path)
    
    # load_cv(processor, tokenizer)
    # load_canto_asr(processor, tokenizer)
    # load_canto_map(processor, tokenizer)

    merge_map_asr()