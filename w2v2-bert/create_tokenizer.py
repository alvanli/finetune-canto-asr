from tqdm import tqdm
import json, glob, re
from datasets import load_dataset, load_from_disk
from collections import Counter

BASE_DIR = "/exp/whisper_yue/"

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]'
punc = re.compile(u'[\u3002\u3005-\u3007\u3010\u3011\uFF0C\uFF1A\uFF1F\uFFE4\uFFE9]')

def remove_special_characters(batch):
    # remove special characters
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    batch["sentence"] = punc.sub('', batch["sentence"])
    return batch

def extract_all_chars(batch):
    all_text = "".join(batch["sentence"])
    # vocab = list(set(all_text))
    return {"vocab": [all_text]}

def load_canto_asr():
    audio_base_path = BASE_DIR + "/cantonese-asr/dataset/data/audio"
    transcript_base_path = BASE_DIR + "/cantonese-asr/dataset/data/transcription"
    all_audio_files = glob.glob(audio_base_path+"/*.wav")
    
    sentences = []
    for audio_file in tqdm(all_audio_files):
        transcript_file = audio_file.replace("audio", "transcription").replace(".wav", ".txt")
        with open(transcript_file, "r", encoding="utf-8") as f:
            content = f.read()
            if len(content.strip()) > 1:
                clean_sentence = re.sub(chars_to_remove_regex, '', content)
                clean_sentence = punc.sub('', clean_sentence)
                sentences.append(clean_sentence)
    return sentences


def load_cantomap():
    input_dir = BASE_DIR + "/CantoMap/processed"
    with open(input_dir+"/annots.json", "r", encoding="utf-8") as f:
        lines = f.read()
        line_dict = json.loads(lines)
    sentences = []
    for key in tqdm(line_dict.keys()):
        curr_val = line_dict[key]
        if len(curr_val.strip()) > 1:
            clean_sentence = re.sub(chars_to_remove_regex, '', curr_val)
            clean_sentence = punc.sub('', clean_sentence)
            sentences.append(clean_sentence)
    return sentences


if __name__ == "__main__":
    ds_train_1 = load_dataset("mozilla-foundation/common_voice_16_0", "yue", split="train", use_auth_token=True)
    ds_train_2 = load_dataset("mozilla-foundation/common_voice_16_0", "zh-HK", split="train", use_auth_token=True)
    ds_test = load_dataset("mozilla-foundation/common_voice_16_0", "yue", split="test", use_auth_token=True)

    vocab_extra = "".join(list(load_canto_asr()+load_cantomap())) + " "
    ds_train_1 = ds_train_1.map(remove_special_characters)
    ds_train_2 = ds_train_2.map(remove_special_characters)
    ds_test = ds_test.map(remove_special_characters)

    vocab_train_1 = ds_train_1.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=ds_train_1.column_names)["vocab"][0]
    vocab_train_2 = ds_train_2.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=ds_train_1.column_names)["vocab"][0]
    vocab_test = ds_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=ds_train_1.column_names)["vocab"][0]

    vocab_list_all = list(set(vocab_train_1) | set(vocab_train_1) | set(vocab_test) | set(vocab_extra))
    counts = Counter(vocab_train_1+vocab_train_1+vocab_test+vocab_extra)
    vocab_list = list(set([char for char, count in counts.items() if count > 15]))
    print(len(vocab_list_all), len(vocab_list))

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('tokenizer/vocab.json', 'w', encoding="utf-8") as vocab_file:
        json.dump(vocab_dict, vocab_file)
