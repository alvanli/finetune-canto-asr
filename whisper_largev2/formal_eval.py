from tqdm import tqdm
import time
import torch
import evaluate 
from datasets import load_dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments

import sys
sys.path.append('/exp/whisper_yue/finetune-whisper-canto')

from normalize_canto import normalize

metric = evaluate.load("cer")
LANGUAGE = "yue"
TASK = "transcribe"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def assisted_generate_with_time(model, inputs, **kwargs):
    start_time = time.time()
    outputs = model.generate(**inputs, **kwargs)
    generation_time = time.time() - start_time
    return outputs, generation_time


if __name__ == "__main__":
    dataset = load_dataset("mozilla-foundation/common_voice_16_0", "yue", split="test", use_auth_token=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    # BASE_WHISPER_MODEL = "openai/whisper-large-v2"
    # model_id = "/exp/whisper_yue/finetune-whisper-canto/whisper_largev2/model_out_01/checkpoint-10000"

    # processor = WhisperProcessor.from_pretrained(BASE_WHISPER_MODEL, language=LANGUAGE, task=TASK)  
    # model = WhisperForConditionalGeneration.from_pretrained(BASE_WHISPER_MODEL, load_in_8bit=False, attn_implementation="sdpa")
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []
    
    # model = PeftModel.from_pretrained(model, model_id)
    # model.eval()
    # model.to(device)


    # BASE_WHISPER_MODEL = "simonl0909/whisper-large-v2-cantonese"
    BASE_WHISPER_MODEL = "alvanlii/whisper-small-cantonese"
    processor = WhisperProcessor.from_pretrained(BASE_WHISPER_MODEL, language=LANGUAGE, task=TASK)  
    model = WhisperForConditionalGeneration.from_pretrained(BASE_WHISPER_MODEL, load_in_8bit=False, attn_implementation="sdpa")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.eval()
    model.to(device)


    all_time = 0
    predictions = []
    references = []

    for sample in tqdm(dataset):
        audio = sample["audio"]
        inputs = processor(audio["array"], sampling_rate=16_000, return_tensors="pt")
        inputs = inputs.to(device=device)
        
        output, gen_time = assisted_generate_with_time(model, inputs, )
        all_time += gen_time
        decoded_output = processor.batch_decode(output, skip_special_tokens=True)[0]
        predictions.append(normalize(decoded_output))
        references.append(normalize(sample["sentence"]))
        print(predictions[-1], references[-1])

        
    print(f"took {all_time} for {len(references)} samples on GPU")
    cer = metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)
    print(cer)
