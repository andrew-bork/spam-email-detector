import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

from tqdm import tqdm
from typing import Callable

from typing import Any

class KaggleDatasets:
    def __init__(self, 
                 data: pd.DataFrame,
                 transforms: dict[str, Any]):
        self.data = data

        for title, transform in transforms.items():
            self.data[title] = [
                transform(x)
                for x in tqdm(self.data["text"])
            ]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: str):
        return PandasColumnDataset(self.data, idx, "target")

class PandasColumnDataset(Dataset):
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


from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

import regex

class KaggleSpamDataset(Dataset):
    def __init__(self, filepath: str, transform=None):
        self.transform = transform
        
        self.data = pd.read_csv(filepath)
        self.data["task_index"] = self.data["target"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = (self.data["text"][idx], self.data["task_index"][idx])

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
        self.data["task_index"] = self.dataset.data["task_index"]

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
    

def tokenize_by_word(s: str):
    return regex.split("\\w+", s.lower())



class Word2VecTransform(torch.nn.Module):
    def __init__(self, model):
        super(EmbeddingTransform, self).__init__()
        self.model = model
        self.model.wv = model

    def forward(self, sample: str):
        tokens = tokenize_by_word(sample)
        vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
        
        if not vecs:
            return torch.zeros(300)

        mat = np.vstack(vecs)
        return torch.as_tensor(mat.mean(axis=0).astype(np.float32))
    
class Word2VecDataset(Dataset):
    def __init__(self, dataset: Dataset, model: Word2Vec | None = None):
        self.dataset = dataset
        
        corpus = [tokenize_by_word(t) for t in self.dataset.data["text"]]

        if(model is None):
            self.model = Word2Vec(
                    sentences=corpus,
                    vector_size=128,
                    window=5,
                    min_count=1,
                    workers=4,
                    epochs=20,
                    sg=1
                )
        else:
            self.model = model
            self.model.wv = model

        self.data = pd.DataFrame()
        self.data["embedding"] = [
            self._embed(x)
            for x in tqdm(self.dataset.data["text"])
        ]
        self.data["task_index"] = self.dataset.data["task_index"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.data.iloc[idx, 0], self.data.iloc[idx, 1])
        return sample

    def _embed(self, x: str):
        tokens = tokenize_by_word(x)
        vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
        
        if not vecs:
            return torch.zeros(300)

        mat = np.vstack(vecs)
        return torch.as_tensor(mat.mean(axis=0).astype(np.float32))


def split_train_val(dataset: Dataset, train_portion:float = 0.8):
    n_samples = len(dataset)
    train_size = int(n_samples * train_portion)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
        
    

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
