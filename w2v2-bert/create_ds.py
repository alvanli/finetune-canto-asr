import glob, os, re, json
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets, Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2BertProcessor, SeamlessM4TFeatureExtractor

from bs4 import BeautifulSoup as bs
import librosa  
import soundfile as sf

BASE_DIR = "/exp/whisper_yue/"
SAMPLING_RATE = 16_000
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]'
punc = re.compile(u'[\u3002\u3005-\u3007\u3010\u3011\uFF0C\uFF1A\uFF1F\uFFE4\uFFE9]')

def remove_special_characters(batch):
    # remove special characters
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    batch["sentence"] = punc.sub('', batch["sentence"])
    return batch


def load_canto_map(processor):
    input_dir = BASE_DIR + "/CantoMap/processed"
    with open(input_dir+"/annots.json", "r", encoding="utf-8") as f:
        lines = f.read()
        line_dict = json.loads(lines)
    info_dict = {
        "input_features": [],
        "input_length": [],
        "labels": []
    }
    for key in tqdm(line_dict.keys()):
        curr_val = line_dict[key]
        if len(curr_val.strip()) > 1:
            arr, rate = librosa.load(f"{input_dir}/{key}.wav", sr=SAMPLING_RATE)
            try:
                feature = processor.feature_extractor(arr, sampling_rate=rate).input_features[0]
                clean_sentence = re.sub(chars_to_remove_regex, '', curr_val)
                clean_sentence = punc.sub('', clean_sentence)
                input_ids = processor(text=clean_sentence).input_ids

                info_dict["input_features"].append(feature),
                info_dict["input_length"].append(len(feature))
                info_dict["labels"].append(input_ids)
            except ValueError:
                print("Bad error")

    ds = Dataset.from_dict(mapping=info_dict)
    ds.save_to_disk("/exp/whisper_yue/data/canto_map")
    return


def load_canto_asr(processor):
    audio_base_path = BASE_DIR + "/cantonese-asr/dataset/data/audio"
    transcript_base_path = BASE_DIR + "/cantonese-asr/dataset/data/transcription"
    all_audio_files = glob.glob(audio_base_path+"/*.wav")
    
    info_dict = {
        "input_features": [],
        "input_length": [],
        "labels": []
    }
    for audio_file in tqdm(all_audio_files):
        transcript_file = audio_file.replace("audio", "transcription").replace(".wav", ".txt")
        with open(transcript_file, "r", encoding="utf-8") as f:
            content = f.read()
        arr, rate = librosa.load(audio_file, sr=SAMPLING_RATE)
        input_length = len(arr) / rate
        if len(content.strip()) > 1:
            feature = processor.feature_extractor(arr, sampling_rate=rate).input_features[0]
            clean_sentence = re.sub(chars_to_remove_regex, '', content)
            clean_sentence = punc.sub('', clean_sentence)
            input_ids = processor(text=clean_sentence).input_ids

            info_dict["input_features"].append(feature),
            info_dict["input_length"].append(len(feature))
            info_dict["labels"].append(input_ids)

    ds = Dataset.from_dict(mapping=info_dict)
    ds.save_to_disk("/exp/whisper_yue/data/canto_asr")
    return


def merge_map_asr():
    ds_canto_asr = load_from_disk("/exp/whisper_yue/data/canto_asr")
    ds_canto_map = load_from_disk("/exp/whisper_yue/data/canto_map")
    ds_train_vect = load_from_disk("/exp/whisper_yue/data/cv_train")
    ds_train_vect_2 = load_from_disk("/exp/whisper_yue/data/cv_train_2")
    big_audio_ds = concatenate_datasets([ds_canto_asr, ds_canto_map, ds_train_vect, ds_train_vect_2])
    big_audio_ds.save_to_disk("/exp/whisper_yue/data/combined_canto_train")


def load_cv(processor):
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(batch["input_features"])
        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    ds_train = load_dataset("mozilla-foundation/common_voice_16_0", "yue", split="train", use_auth_token=True)  
    ds_train_2 = load_dataset("mozilla-foundation/common_voice_16_0", "zh-HK", split="train", use_auth_token=True)  
    ds_test = load_dataset("mozilla-foundation/common_voice_16_0", "yue", split="test", use_auth_token=True)

    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16_000))
    ds_train_2 = ds_train_2.cast_column("audio", Audio(sampling_rate=16_000))
    ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16_000))

    ds_train = ds_train.map(remove_special_characters)
    ds_train_2 = ds_train_2.map(remove_special_characters)
    ds_test = ds_test.map(remove_special_characters)

    ds_train = ds_train.map(prepare_dataset, remove_columns=ds_train.column_names)
    ds_train_2 = ds_train_2.map(prepare_dataset, remove_columns=ds_train_2.column_names)
    ds_test = ds_test.map(prepare_dataset, remove_columns=ds_test.column_names)

    ds_train.save_to_disk("/exp/whisper_yue/data/cv_train")
    ds_train_2.save_to_disk("/exp/whisper_yue/data/cv_train_2")
    ds_test.save_to_disk("/exp/whisper_yue/data/cv_test")
    return


if __name__ == "__main__":
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./tokenizer", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=SAMPLING_RATE, padding_value=0.0)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    load_cv(processor)
    load_canto_asr(processor)
    load_canto_map(processor)

    merge_map_asr()