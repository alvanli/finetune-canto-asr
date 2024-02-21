import csv

from transformers import WhisperProcessor
from datasets import load_from_disk

max_label_length = 128




if __name__ == "__main__":
    eval_preds = {}
    with open('./output/transcription.csv', mode ='r') as f:
        csv_file = csv.DictReader(f)
        next(csv_file)
        for line in csv_file:
            file_id = line["file_id"]
            preds = line["whisper_transcript"]
            eval_preds[file_id] = preds

    processor = WhisperProcessor.from_pretrained("simonl0909/whisper-large-v2-cantonese")

    def prepare_dataset(batch):
        input_labels = batch["labels"]
        batch["labels"] = input_labels if len(input_labels) < max_label_length else input_labels[:max_label_length]
        f_id = processor.tokenizer.decode(batch["file_id"])
        batch["whisper_transcript"] = eval_preds[f_id] if f_id in eval_preds.keys() else "" 
        return batch

    def remove_blank(batch):
        if batch["whisper_transcript"] == "":
            return False
        return True

    raw_datasets = load_from_disk("/exp/whisper_yue/whisper_data/combined_canto_train")
    raw_datasets = raw_datasets.map(prepare_dataset)
    print(len(raw_datasets))
    raw_datasets = raw_datasets.filter(remove_blank)
    print(len(raw_datasets))
    raw_datasets.save_to_disk("./output/pseudo_preds")