import re
import time
from tqdm import tqdm
import torch
import evaluate
from datasets import load_from_disk, load_dataset

from transformers import pipeline, Wav2Vec2CTCTokenizer, Wav2Vec2BertProcessor, SeamlessM4TFeatureExtractor

metric = evaluate.load("cer")

import sys
sys.path.append('/exp/whisper_yue/finetune-whisper-canto')

from normalize_canto import normalize

# def compute_metrics(pred):
#     pred_logits = pred.predictions
#     pred_ids = np.argmax(pred_logits, axis=-1)

#     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

#     pred_str = processor.batch_decode(pred_ids)
#     # we do not want to group tokens when computing the metrics
#     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

#     cer = metric.compute(predictions=pred_str, references=label_str)

#     return {"cer": cer}

def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["sentence"]}

if __name__ == "__main__":
    ds_test_vect = load_dataset("mozilla-foundation/common_voice_16_0", "yue", split="test", use_auth_token=True)
    # repo_name = "alvanlii/whisper-small-cantonese"
    repo_name = "/exp/whisper_yue/finetune-whisper-canto/w2v2-bert/canto_bertw2v2/checkpoint-6000"
    # repo_name = "simonl0909/whisper-large-v2-cantonese"
    # repo_name = "CAiRE/wav2vec2-large-xlsr-53-cantonese"
    # repo_name = "alvanlii/wav2vec2-BERT-cantonese"

    bert_asr = pipeline(
        "automatic-speech-recognition", model=repo_name, device="cuda"
    )
    predictions, references = [], []

    count = 0
    start_time = time.time()
    # run streamed inference
    for out in bert_asr(data(ds_test_vect), batch_size=1):
        count += 1
        predictions.append(normalize(out["text"]))
        references.append(normalize(out["reference"][0]))
        print(f"{predictions[-1]}---{references[-1]}")
    end_time = time.time()
    cer = metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)
    print(cer)
    print((end_time-start_time), count)
