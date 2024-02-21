from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2BertProcessor, SeamlessM4TFeatureExtractor

SAMPLING_RATE = 16_000

if __name__ == "__main__":
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./tokenizer", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=SAMPLING_RATE, padding_value=0.0)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    print(processor(text=["在烏克蘭與俄羅斯邊境附近的別爾哥羅德州發生墜機事件後"]).input_ids)

