import time
import numpy as np

import torch
import torch.nn as nn
from datasets import load_from_disk, concatenate_datasets
from augment.pt_augs import get_spectrogram, do_time_stretch, do_freq_masking, do_time_masking


def process_function(datum):
    aug_feat = torch.tensor(datum["input_features"])
    aug_feat = torch.unsqueeze(aug_feat, 2)
    if np.random.random() < 0.5:
        chances = np.random.random()
        if chances < 0.3:
            aug_feat = do_time_masking(aug_feat)
        if chances > 0.2:
            aug_feat = do_time_stretch(aug_feat)
        if 0.3 < chances < 0.7:
            aug_feat = do_freq_masking(aug_feat)
    aug_feat = aug_feat.squeeze()

    aug_feat = aug_feat.tolist()
    datum["input_features"] = aug_feat
    return datum


if __name__ == "__main__":
    ds = load_from_disk("/exp/whisper_yue/data/combined_canto_train")

    dses = []
    for _ in range(3):
        dses.append(ds.map(
            lambda x: process_function(x), 
        ))
        ds.cleanup_cache_files()
    
    big_boss_ds = concatenate_datasets(dses)
    big_boss_ds.save_to_disk("/exp/whisper_yue/data/combined_canto_train_augment")

