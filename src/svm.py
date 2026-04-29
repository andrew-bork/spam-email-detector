from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)


import torch
from tqdm import tqdm

class SVM:
    def __init__(self, **kwargs):
        self.model = svm.SVC(**kwargs)
    
    def forward(self, x: torch.Tensor):
        return self.model.predict(x)

    def train(self, train_loader, val_loader):
        for (X_batch, y_batch) in tqdm(train_loader):
            self.model.partial_fit(X_batch, y_batch)
        return [], []