import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from typing import Iterable


class Trainer:
    def train(self, train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]], val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]], **kwargs) -> tuple[list[float], list[float]]:
        raise NotImplemented()
    
    def evaluate(self, data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, any]:
        raise NotImplemented()
    
    def build_loaders(self, train_dataset, val_dataset, batch_size=1, **kwargs) -> tuple[Iterable[tuple[torch.Tensor, torch.Tensor]], Iterable[tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplemented()

class SklearnTrainer(Trainer):
    def __init__(self, model):
        self.model = model

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def build_loaders(self, train_dataset, val_dataset, **kwargs):
        return [train_dataset], [val_dataset]

    def train(self, train_loader, val_loader, **kwargs):
        for (X_batch, y_batch) in tqdm(train_loader):
            self.model.partial_fit(X_batch, y_batch)
        return [], []
    
    def evaluate(self, data_loader):
        all_preds = []
        all_targets = []

        for X_batch, y_batch in tqdm(data_loader):
            outputs = self.forward(X_batch.numpy())
            predictions = np.argmax(outputs, dim=1)
            all_preds.append(predictions)
            all_targets.append(y_batch.numpy())
        
        all_preds = np.concat(all_preds)
        all_targets = np.concat(all_targets)
        
        accuracy = (all_preds == all_targets).sum() / len(all_preds)
        f1 = f1_score(all_targets, all_preds, average="macro")
        
        
        metrics = {
            'loss': 0,
            'accuracy': accuracy,
            'f1_macro': f1
        }
        
        return metrics
    

class TorchTrainer(Trainer):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def build_loaders(self, train_dataset, val_dataset, batch_size=1, **kwargs):
        return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def train(self, train_loader, val_loader, device=None, epochs=100, lr=0.1, weight_decay=0.01, **kwargs):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            total = 0
            epoch_loss = 0.0
            model.train()
            for (X_batch, y_batch) in tqdm(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total += len(y_batch)
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            model.eval()
            total = 0
            total_correct = 0
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
                    predictions = torch.argmax(outputs, dim=1)
                    total_correct += (predictions == y_batch).sum().item()
                    total += len(y_batch)
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {100*total_correct/total:.2f}%")
        
        return train_losses, val_losses
