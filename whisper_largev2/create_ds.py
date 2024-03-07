import glob, os, re, json
from tqdm import tqdm
from multiprocessing import Process
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets, Audio
from transformers import WhisperProcessor

from bs4 import BeautifulSoup as bs
import librosa  
import soundfile as sf

BASE_DIR = "/exp/whisper_yue/"
SAVE_DIR = "/exp/whisper_yue/whisper_data"

SAMPLING_RATE = 16_000
SPLIT_LENGTH = 2000
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", task="transcribe")

def load_canto_map():
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
    length_total_seconds = 0

    for idx, key in enumerate(list(line_dict.keys())):
        curr_val = line_dict[key]
        if len(curr_val.strip()) > 3:
            arr, rate = librosa.load(f"{input_dir}/{key}.wav", sr=SAMPLING_RATE)
            length_seconds = librosa.get_duration(y=arr, sr=rate)
            try:
                feature = processor.feature_extractor(arr, sampling_rate=rate).input_features[0]
                clean_sentence = curr_val
                input_ids = processor(text=clean_sentence).input_ids

                info_dict["input_features"].append(feature),
                info_dict["input_length"].append(len(arr) / rate)
                info_dict["labels"].append(input_ids)
                length_total_seconds += length_seconds
            except ValueError:
                print("Bad error")
        if idx % SPLIT_LENGTH == 0 and idx != 0:
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

    print("="*10)
    print(f"Finished loading CantoMap dataset")
    print(f"Length: {length_total_seconds} seconds")
    print("="*10)
    return


def load_canto_asr():
    audio_base_path = BASE_DIR + "/cantonese-asr/dataset/data/audio"
    transcript_base_path = BASE_DIR + "/cantonese-asr/dataset/data/transcription"
    all_audio_files = glob.glob(audio_base_path+"/*.wav")
    
    info_dict = {
        "input_features": [],
        "input_length": [],
        "labels": []
    }
    count_idx = 1
    length_total_seconds = 0
    for idx, audio_file in enumerate(all_audio_files):  
        transcript_file = audio_file.replace("audio", "transcription").replace(".wav", ".txt")
        with open(transcript_file, "r", encoding="utf-8") as f:
            content = f.read()
        arr, rate = librosa.load(audio_file, sr=SAMPLING_RATE)
        length_seconds = librosa.get_duration(y=arr, sr=rate)
        input_length = len(arr) / rate
        if len(content.strip()) > 3:
            feature = processor.feature_extractor(arr, sampling_rate=rate).input_features[0]
            clean_sentence = content
            input_ids = processor(text=clean_sentence).input_ids

            info_dict["input_features"].append(feature)
            info_dict["input_length"].append(input_length)
            info_dict["labels"].append(input_ids)
            length_total_seconds += length_seconds

        if idx % SPLIT_LENGTH == 0 and idx != 0:
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

    print("="*10)
    print(f"Finished loading Cantonse ASR")
    print(f"Length: {length_total_seconds} seconds")
    print("="*10)
    return


def merge_everything():
    ds_canto_asr = [load_from_disk(dir) for dir in glob.glob(f"{SAVE_DIR}/canto_asr*")]
    ds_canto_map = [load_from_disk(dir) for dir in glob.glob(f"{SAVE_DIR}/canto_map*")]
    ds_canto_pseudo = [load_from_disk(dir) for dir in glob.glob(f"{SAVE_DIR}/canto_pseudo*")]
    ds_train_vect = load_from_disk(f"{SAVE_DIR}/cv_train")
    ds_train_vect_2 = load_from_disk(f"{SAVE_DIR}/cv_train_2")
    big_audio_ds = concatenate_datasets(ds_canto_asr + ds_canto_map + ds_canto_pseudo + [ds_train_vect, ds_train_vect_2])
    big_audio_ds.save_to_disk(f"{SAVE_DIR}/combined_canto_train")


def load_cv():
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


def load_pseudo_ds():
    all_audio_files = sorted(glob.glob(f"{BASE_DIR}/canto-youtube-dl/audio/*.mp3") + glob.glob(f"{BASE_DIR}/sbs_cantonese/audio/*.flac"))
    info_dict = {
        "input_features": [],
        "input_length": [],
        "labels": []
    }
    count_idx = 1
    length_total_seconds = 0
    for idx, audio_file in enumerate(all_audio_files):  
        transcript_file = audio_file.replace("audio", "clean").replace(".mp3", ".txt").replace(".flac", ".txt")

        if not os.path.isfile(transcript_file):
            continue

        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read()
        arr, rate = librosa.load(audio_file, sr=SAMPLING_RATE)
        length_seconds = librosa.get_duration(y=arr, sr=rate)
        input_length = len(arr) / rate
        if len(transcript.strip()) > 5:
            feature = processor.feature_extractor(arr, sampling_rate=rate).input_features[0]
            clean_sentence = transcript
            input_ids = processor(text=clean_sentence).input_ids

            info_dict["input_features"].append(feature)
            info_dict["input_length"].append(input_length)
            info_dict["labels"].append(input_ids)
            length_total_seconds += length_seconds

        if idx % SPLIT_LENGTH == 0 and idx != 0:
            ds = Dataset.from_dict(mapping=info_dict)
            ds.save_to_disk(f"{SAVE_DIR}/canto_pseudo_{count_idx}")
            count_idx += 1
            info_dict = {
                "input_features": [],
                "input_length": [],
                "labels": []
            }    

    ds = Dataset.from_dict(mapping=info_dict)
    ds.save_to_disk(f"{SAVE_DIR}/canto_pseudo_end")

    print("="*10)
    print(f"Finished loading pseudo dataset")
    print(f"Length: {length_total_seconds} seconds")
    print("="*10)
    return

def load_others():
    # return
    load_cv()
    # load_canto_asr()
    # load_canto_map()

if __name__ == "__main__":
    p1 = Process(target=load_others)
    p1.start()

    p2 = Process(target=load_pseudo_ds)
    p2.start()

    # load_cv()
    # load_canto_asr()
    # load_canto_map()
    # load_pseudo_ds()


    p1.join()
    p2.join()
    merge_everything()