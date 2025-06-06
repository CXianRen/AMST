import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import sys

class Mock_dataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = idx % 6
        sid = "0" 
        audio_feature = torch.zeros((1024, 128))
        image_n = torch.zeros((3, 3, 224, 224))
        return audio_feature, image_n, label, sid

class Mock_3m_dataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = idx % 2
        sid = "0" 
        audio_feature = torch.zeros((1024, 128))
        image_n = torch.zeros((3, 3, 224, 224))
        
        tokenizer = torch.zeros(128, dtype=torch.long)
        padding_mask = torch.zeros(128, dtype=torch.long)
    
        return tokenizer, padding_mask, image_n, audio_feature, label, sid

def run_trainer_test(trainer_class, test_name, args=None):
    # disable the printing 
    # sys.stdout = open(os.devnull, 'w')
    if args is None:
        trainer = trainer_class()
    else:
        trainer = trainer_class(args)
    trainer.train_validate()
    # enable the printing
    sys.stdout = sys.__stdout__
    print(f"passed {test_name}")