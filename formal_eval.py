from tqdm import tqdm
import time
import torch
import evaluate 
from datasets import load_dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCTC, Wav2Vec2BertProcessor, Wav2Vec2Processor
from peft import PeftConfig, PeftModel
import librosa

import sys
sys.path.append('/exp/whisper_yue/finetune-whisper-canto')

from normalize_canto import normalize

NORMALIZE = True
IS_WAV2VEC = False
IS_LORA = False
USE_GPU = True
LANGUAGE = 'zh'

# model_id = "alvanlii/wav2vec2-BERT-cantonese"
# model_id = "simonl0909/whisper-large-v2-cantonese"
# model_id = 'CAiRE/wav2vec2-large-xlsr-53-cantonese'
# model_id = 'alvanlii/whisper-largev2-cantonese-peft-lora'
# model_id = 'Oblivion208/whisper-large-v2-lora-cantonese'
# model_id = '/exp/whisper_yue/finetune-whisper-canto/whisper_largev2/checkpoint-6500'
model_id = 'openai/whisper-large-v3'
model_id = 'Scrya/whisper-large-v2-cantonese'
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
        # processor = Wav2Vec2Processor.from_pretrained(model_id)
        processor = Wav2Vec2BertProcessor.from_pretrained(model_id)
    elif IS_LORA:
        processor = None
    else:
        processor = AutoProcessor.from_pretrained(model_id, language=LANGUAGE)

    if IS_WAV2VEC:
        model = AutoModelForCTC.from_pretrained(
            model_id,
            torch_dtype=torch_dtype
        )
        model.to(device)
    elif IS_LORA:
        peft_config = PeftConfig.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto", attn_implementation="sdpa",
             low_cpu_mem_usage=True, torch_dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(model, model_id)
        processor = AutoProcessor.from_pretrained(peft_config.base_model_name_or_path)
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa"
        )
        # model.generation_config.language = LANGUAGE
        model.to(device)


    all_time = 0
    total_seconds = 0
    predictions = []
    references = []

    for sample in tqdm(dataset):
        audio = sample["audio"]
        length_seconds = librosa.get_duration(y=audio['array'], sr=16_000)
        total_seconds += length_seconds
        inputs = processor(audio["array"], sampling_rate=16_000, return_tensors="pt")
        inputs = inputs.to(device=device, dtype=torch_dtype)
        
        if IS_WAV2VEC:
            features = inputs.input_features
            # features = inputs.input_values
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
        
    print(f"took {all_time} for {len(references)} ({total_seconds}s) samples on GPU")
    cer = metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)
    print(cer)
