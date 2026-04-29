import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

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
            sample = self.transform.forward(sample)

        return sample

# class DatasetTransformer(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
            
#         sample = (self.data.iloc[idx, 0], self.data.iloc[idx, 1])

#         if self.transform:
#             sample = self.transform.forward(sample)

#         return sample


class TokenizerTransform(torch.nn.Module):
    def __init__(self, tokenizer):
        super(TokenizerTransform, self).__init__()
        self.tokenizer = tokenizer

    def _tokenize_to_ids(self, s: str):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s))

    def forward(self, sample):
        texts, labels = sample
        # if(type(texts) == str):
        return self._tokenize_to_ids(texts), labels
        # return (map(self._tokenize_to_ids, texts), labels)
    
class ToTensor(torch.nn.Module):
    def __init__(self):
        super(ToTensor, self).__init__()

    def forward(self, sample):
        texts, labels = sample
        # if(type(texts) == str):
        return torch.as_tensor(texts), torch.as_tensor(labels)
        # return list(map(torch.as_tensor, texts)), torch.as_tensor(labels)
    
    

if __name__ == "__main__":
    VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
    dataset = KaggleSpamDataset("./data/emails.csv",
                                transform=torch.nn.Sequential(TokenizerTransform(AutoTokenizer.from_pretrained("bert-base-cased")), ToTensor()))
    a,b = dataset[[0,1]]
    print(list(a), b)