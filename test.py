import transformers
from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("mozilla-foundation/common_voice_8_0", "zh-HK", use_auth_token=True, streaming=True)
    ds_1 = load_dataset("mozilla-foundation/common_voice_10_0", "zh-HK", split="train+validation", use_auth_token=True)
    print(ds["train"].features)