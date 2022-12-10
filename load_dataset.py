import numpy
import shutil
import glob, os, re, json
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from transformers import WhisperProcessor

from bs4 import BeautifulSoup as bs
import librosa  
import soundfile as sf
# from scipy.io.wavfile import read as wavread

BASE_DIR = "/exp/whisper_yue"

def process_annot(words):
    # words = re.sub(r'(&amp;)', '', words)
    words = words.replace("#",'')
    words = words.replace("&",'')
    words = re.sub(r'[a-z]{1,5}[0-9]','', words)
    words = words.replace("xxx", "")
    return words

def process_canto_map():
    all_audio_files = glob.glob(BASE_DIR+"/CantoMap/ConversationData/*/*.wav")
    output_dir = BASE_DIR + "/CantoMap/processed"

    annot_map = dict()
    for audio_file in tqdm(all_audio_files):
        annot_file = audio_file.replace(".wav", ".eaf")
        file_name = os.path.basename(audio_file).split(".")[0]
        if os.path.exists(annot_file):
            with open(annot_file, "r") as f:
                annot_content = f.read()
            arr, rate = librosa.load(audio_file, sr=16000)

            soup = bs(annot_content, "xml")
            time_slots = soup.find_all("TIME_SLOT")
            slot_maps = dict({time_slot["TIME_SLOT_ID"]: int(time_slot["TIME_VALUE"]) for time_slot in time_slots})

            annots = soup.find_all("ALIGNABLE_ANNOTATION")
            for annot in annots:
                annot_text = annot.find_all("ANNOTATION_VALUE")[0].text
                processed_annot = process_annot(annot_text)
                if process_annot:
                    aid = annot["ANNOTATION_ID"]
                    ref1 = slot_maps[annot["TIME_SLOT_REF1"]]
                    ref2 = slot_maps[annot["TIME_SLOT_REF2"]]
                    start_idx = int(rate * ref1 / 1000)
                    stop_idx = int(rate * ref2 / 1000)
                    sf.write(output_dir + f"/{file_name}_{aid}.wav", arr[start_idx:stop_idx], rate)
                    annot_map[f"{file_name}_{aid}"] = processed_annot
    with open(output_dir+"/annots.json", "w", encoding="utf-8") as f:
        json.dump(annot_map, f, ensure_ascii=False)

    return


def process_canto_map_2():
    all_audio_files = glob.glob(BASE_DIR+"/CantoMap/ConversationData/*/*.wav")
    output_dir = BASE_DIR + "/CantoMap/hf_ds/"

    with open(BASE_DIR+"/CantoMap/hf_ds/metadata.csv", "a") as meta_f:
        meta_f.write("file_name,transcription\n")

    annot_map = dict()
    for audio_file in tqdm(all_audio_files):
        annot_file = audio_file.replace(".wav", ".eaf")
        file_name = os.path.basename(audio_file).split(".")[0]
        if os.path.exists(annot_file):
            with open(annot_file, "r") as f:
                annot_content = f.read()
            arr, rate = librosa.load(audio_file, sr=16000)

            soup = bs(annot_content, "xml")
            time_slots = soup.find_all("TIME_SLOT")
            slot_maps = dict({time_slot["TIME_SLOT_ID"]: int(time_slot["TIME_VALUE"]) for time_slot in time_slots})

            annots = soup.find_all("ALIGNABLE_ANNOTATION")
            for annot in annots:
                annot_text = annot.find_all("ANNOTATION_VALUE")[0].text
                processed_annot = process_annot(annot_text)
                if len(processed_annot.strip())>1:
                    aid = annot["ANNOTATION_ID"]
                    ref1 = slot_maps[annot["TIME_SLOT_REF1"]]
                    ref2 = slot_maps[annot["TIME_SLOT_REF2"]]
                    start_idx = int(rate * ref1 / 1000)
                    stop_idx = int(rate * ref2 / 1000)
                    sf.write(output_dir + f"data/{file_name}_{aid}.wav", arr[start_idx:stop_idx], rate)
                    with open(BASE_DIR+"/CantoMap/hf_ds/metadata.csv", "a") as meta_f:
                        meta_f.write(f"data/{file_name}_{aid}.wav,{processed_annot}\n")

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
            arr, rate = librosa.load(input_dir + f"/{key}.wav", sr=16000)
            info_dict["input_features"].append(processor.feature_extractor(arr, sampling_rate=rate).input_features[0]),
            info_dict["input_length"].append( len(arr) / rate)
            info_dict["labels"].append(processor.tokenizer(curr_val).input_ids)

    ds = Dataset.from_dict(mapping=info_dict)
    ds.save_to_disk("/exp/hf_ds/canto_map")

def load_canto_asr(processor):
    audio_base_path = BASE_DIR + "/cantonese-asr/dataset/zippy/audio"
    transcript_base_path = BASE_DIR + "/cantonese-asr/dataset/zippy/transcription"
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
        arr, rate = librosa.load(audio_file, sr=16000)
        input_length = len(arr) / rate
        if input_length < 35 and len(content.strip()) > 1:
            info_dict["input_features"].append(processor.feature_extractor(arr, sampling_rate=rate).input_features[0]),
            info_dict["input_length"].append(input_length)
            info_dict["labels"].append(processor.tokenizer(content).input_ids)

    ds = Dataset.from_dict(mapping=info_dict)
    ds.save_to_disk("/exp/hf_ds/canto_asr")
    return

def load_canto_asr_2():
    audio_base_path = BASE_DIR + "/cantonese-asr/dataset/zippy/audio"
    transcript_base_path = BASE_DIR + "/cantonese-asr/dataset/zippy/transcription"
    all_audio_files = glob.glob(audio_base_path+"/*.wav")
    with open("/exp/hf_ds/canto_asr/metadata.csv", "w") as meta_f:
        meta_f.write("file_name,transcription\n")

    for audio_file in tqdm(all_audio_files):
        transcript_file = audio_file.replace("audio", "transcription").replace(".wav", ".txt")
        with open(transcript_file, "r", encoding="utf-8") as f:
            content = f.read()
        file_name = os.path.basename(audio_file).split(".")[0]
        shutil.copyfile(audio_file, f"/exp/hf_ds/canto_asr/data/{file_name}.wav")
        with open("/exp/hf_ds/canto_asr/metadata.csv", "a") as meta_f:
            meta_f.write(f"data/{file_name}.wav,{content}\n")

def prepare_dataset(batch):
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    transcription = batch["transcription"]
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch


def merge_map_asr():
    ds_canto_asr = load_dataset("audiofolder", data_dir="/exp/hf_ds/canto_asr")
    ds_canto_map = load_dataset("audiofolder", data_dir="/exp/hf_ds/canto_map")
    ds_train_vect = load_from_disk("/exp/hf_ds/processed_common_11_hk/train")
    # print(ds_canto_asr)
    # print(ds_canto_map)
    print(type(ds_train_vect))
    two_audio_ds = concatenate_datasets([ds_canto_asr["train"], ds_canto_asr['train']]).map(prepare_dataset).with_format("torch")
    big_audio_ds = concatenate_datasets([ds_train_vect, two_audio_ds])
    big_audio_ds.save_to_disk("/exp/hf_ds/combined_canto")


def load_cancor():
    return

def load_mdcc():
    return

if __name__ == "__main__":
    merge_map_asr()
    # process_canto_map_2()
    # load_canto_asr_2()
    # processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")
    # load_canto_asr(processor)
    # load_canto_map(processor)