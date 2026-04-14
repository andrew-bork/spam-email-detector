import torch
from torch.utils.data import Dataset
import csv
import pandas as pd
import numpy as np
from torchtext.transforms import BERTTokenizer

class KaggleSpamDataset(Dataset):
    def __init__(self, filepath: str, transform=None):
        self.transform = transform
        
        self.data = pd.read_csv(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = (self.data.iloc[idx, 0], self.data.iloc[idx, 1])

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
    dataset = KaggleSpamDataset("./data/emails.csv",
                                transform=BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True))
    a,b = dataset[0]
    print(a, b)