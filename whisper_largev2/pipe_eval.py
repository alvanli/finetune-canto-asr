from tqdm import tqdm
import time, re
import torch
import evaluate 
from datasets import load_dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, pipeline


metric = evaluate.load("cer")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
punc = re.compile(u'[\u3002\u3005-\u3007\u3010\u3011\uFF0C\uFF1A\uFF1F\uFFE4\uFFE9]')


def assisted_generate_with_time(model, inputs, **kwargs):
    start_time = time.time()
    outputs = model.generate(**inputs, **kwargs)
    generation_time = time.time() - start_time
    return outputs, generation_time


def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["sentence"]}


if __name__ == "__main__":
    dataset = load_dataset("mozilla-foundation/common_voice_16_0", "yue", split="test", use_auth_token=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    BASE_WHISPER_MODEL = "openai/whisper-large-v3"
    LANGUAGE = "yue"
    TASK = "transcribe"

    model_id = "/exp/whisper_yue/finetune-whisper-canto/whisper_largev2/model_out_01/checkpoint-3500"

    # processor = WhisperProcessor.from_pretrained(BASE_WHISPER_MODEL, language=LANGUAGE, task=TASK)  
    # model = WhisperForConditionalGeneration.from_pretrained(BASE_WHISPER_MODEL, load_in_8bit=False, use_flash_attention_2=True)
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []

    # model = PeftModel.from_pretrained(model, model_id)

    # config = LoraConfig(use_dora=True, r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()
    # model.config.use_cache = False
    
    # model.to(device)

    asr = pipeline(
        "automatic-speech-recognition", model=model_id, device="cuda",
        chunk_length_s=20, return_timestamps=True
    )

    all_time = 0
    predictions = []
    references = []

    for datum in tqdm(dataset):
        out = asr(inputs=datum['audio'])
        pred = out["text"]
        clean_sentence = datum["sentence"]
        clean_sentence = punc.sub('', clean_sentence)
        pred = punc.sub('', pred)


        predictions.append(pred)
        references.append(clean_sentence)

        # print(predictions[-1], references[-1])

        # audio = sample["audio"]
        # inputs = processor(audio["array"], sampling_rate=16_000, return_tensors="pt")
        # inputs = inputs.to(device=device)
        
        # output, gen_time = assisted_generate_with_time(model, inputs, )
        # all_time += gen_time
        # decoded_output = processor.batch_decode(output, skip_special_tokens=True)[0]
        # predictions.append(decoded_output)
        # print(sample['sentence'], decoded_output)
        # references.append(sample["sentence"])
        
    print(f"took {all_time} for {len(references)} samples on GPU")
    cer = metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)
    print(cer)
