import wget 
import librosa
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer, WhisperTokenizer, WhisperProcessor

peft_model_id = "alvanlii/whisper-largev2-cantonese-peft-lora"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)

task = "transcribe"
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, task=task)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, task=task)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

wget.download("https://datasets-server.huggingface.co/cached-assets/mozilla-foundation/common_voice_13_0/--/8c6ec32c340031124236862abfed7be1583d1172/--/zh-HK/test/3/audio/audio.mp3")
y, sr = librosa.load('audio.mp3')
audio = y
text = pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
