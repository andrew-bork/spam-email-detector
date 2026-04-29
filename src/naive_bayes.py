from sklearn.naive_bayes import GaussianNB
import torch
from tqdm import tqdm

class NaiveBayes:
    def __init__(self):
        self.model = GaussianNB()
    
    def forward(self, x: torch.Tensor):
        return self.model.predict(x)

    def train(self, train_loader, val_loader):
        for (X_batch, y_batch) in tqdm(train_loader):
            self.model.partial_fit(X_batch, y_batch)
        return [], []