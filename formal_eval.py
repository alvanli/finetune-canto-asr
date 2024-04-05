from tqdm import tqdm
import time
import torch
import evaluate 
from datasets import load_dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCTC, Wav2Vec2BertProcessor


import sys
sys.path.append('/exp/whisper_yue/finetune-whisper-canto')

from normalize_canto import normalize

NORMALIZE = False
IS_WAV2VEC = True
USE_GPU = False
model_id = "alvanlii/wav2vec2-BERT-cantonese"

metric = evaluate.load("cer")

if USE_GPU:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
else:
    device = "cpu"
    torch_dtype = torch.float32


def generate_with_time(model, inputs, **kwargs):
    start_time = time.time()
    outputs = model.generate(**inputs, **kwargs)
    generation_time = time.time() - start_time
    return outputs, generation_time


def w2v_pedict(model, inputs):
    start_time = time.time()
    with torch.no_grad():
        logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    generation_time = time.time() - start_time
    return predicted_ids, generation_time


if __name__ == "__main__":
    dataset = load_dataset("mozilla-foundation/common_voice_16_0", "yue", split="test", use_auth_token=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    if IS_WAV2VEC:
        processor = Wav2Vec2BertProcessor.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id)

    if IS_WAV2VEC:
        model = AutoModelForCTC.from_pretrained(
            model_id,
            torch_dtype=torch_dtype
        )
        model.to(device)
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        model.to(device)


    all_time = 0
    predictions = []
    references = []

    for sample in tqdm(dataset):
        audio = sample["audio"]
        inputs = processor(audio["array"], sampling_rate=16_000, return_tensors="pt")
        inputs = inputs.to(device=device, dtype=torch_dtype)
        
        if IS_WAV2VEC:
            features = inputs.input_features
            output, gen_time = w2v_pedict(model, features)
        else:
            output, gen_time = generate_with_time(model, inputs)

        all_time += gen_time
        pred = processor.batch_decode(output, skip_special_tokens=True, group_tokens=False)[0]

        if NORMALIZE:
            pred = normalize(pred)
        predictions.append(pred)
        
        ground_truth = sample["sentence"]

        if NORMALIZE:
            ground_truth = normalize(ground_truth)

        references.append(ground_truth)
        
        print(f"{predictions[-1]}==={references[-1]}")
        
    print(f"took {all_time} for {len(references)} samples on GPU")
    cer = metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)
    print(cer)
