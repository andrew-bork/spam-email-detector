from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
from tqdm import tqdm

class NearestNeighbor:
    def __init__(self, **kwargs):
        self.model = NearestNeighbors(**kwargs)
    
    def forward(self, x: torch.Tensor):
        return self.model.predict(x)

    def train(self, train_loader, val_loader):
        for (X_batch, y_batch) in tqdm(train_loader):
            self.model.partial_fit(X_batch, y_batch)
        return [], []
    

