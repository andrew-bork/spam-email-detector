import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

from tqdm import tqdm
from typing import Callable
class KaggleDatasets:
    def __init__(self, 
                 filepath: str,
                 transforms: dict[str, any]):
        self.data = pd.read_csv(filepath)

        for title, transform in transforms.items():
            self.data[title] = [
                transform(x)
                for x in tqdm(self.dataset.data["text"])
            ]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: str):
        return PandasColumnDataset(self.data, idx, "target")

class PandasColumnDataset:
    def __init__(self, data:pd.DataFrame, input_column:str, output_column: str = "target"):
        self.data = data
        self.input_column = input_column
        self.output_column = output_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = (self.data[self.input_column][idx], self.data[self.output_column][idx])

        return sample


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

class EmbeddingDataset(Dataset):
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.data = pd.DataFrame()
        self.data["embedding"] = [
            transform(x)
            for x in tqdm(self.dataset.data["text"])
        ]
        self.data["target"] = self.dataset.data["target"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.data.iloc[idx, 0], self.data.iloc[idx, 1])
        return sample
class EmbeddingTransform(torch.nn.Module):
    def __init__(self, model):
        super(EmbeddingTransform, self).__init__()
        self.model = model
    
    def forward(self, sample):
        return self.model.encode(sample)

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
        return torch.as_tensor(texts), torch.as_tensor(labels)
    
    

if __name__ == "__main__":
    # VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
    dataset = KaggleSpamDataset("./data/emails.csv",
                                # transform=torch.nn.Sequential(TokenizerTransform(AutoTokenizer.from_pretrained("bert-base-cased")), ToTensor())
                                )
    
    # a,b = dataset[[0,1]]
    # print(list(a), b)

    from sentence_transformers import SentenceTransformer

    sentence_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    dataset = EmbeddingDataset(dataset,                   
            transform=EmbeddingTransform(sentence_embedding_model)
        )
    
    
    # a,b = dataset[[0,1]]
    # print(list(a), b)
