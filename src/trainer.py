import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Any
import time

from pydantic import BaseModel

from dataset_loader import split_train_val

class EvaluationMetrics(BaseModel):
    loss: float
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float

class TrainingMetrics(BaseModel):
    training_metrics: EvaluationMetrics
    validation_metrics: EvaluationMetrics
    
    training_time: float
    inference_time: float
    
    training_losses: list[float]
    validation_losses: list[float]


class ModelOptions(BaseModel):
    input_size: int
    hidden_size: int
    output_size: int



class TrainingOptions(BaseModel):
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    patience: int = 3

from typing import TypeVar, Generic

T = TypeVar('T')
M = TypeVar('M')

class Trainer(Generic[T, M]):
    def __init__(self, model: Any):
        raise NotImplemented()
    
    def train(self, train_loader: torch.utils.data.DataLoader[T], val_loader: torch.utils.data.DataLoader[T], options: TrainingOptions) -> tuple[list[float], list[float]]:
        raise NotImplemented()
    
    def inference(self, x: np.ndarray) -> bool:
        raise NotImplemented()
    
    def timed_inference(self, x: np.ndarray) -> tuple[bool, float]:
        start_time = time.perf_counter()
        result = self.inference(x)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return (result, total_time)
    
    def run(self, dataset: torch.utils.data.Dataset[T], options: TrainingOptions) -> TrainingMetrics:
        train_loader, validation_loader = self.build_loaders(dataset, options)

        # print(f"\nTraining {title}...")
        start = time.perf_counter()
        training_losses, validation_losses = self.train(train_loader, validation_loader, options)
        end = time.perf_counter()
        train_time = end - start

        # print(f"\nEvaluating {title}...")
        start = time.perf_counter()
        train_metrics = self.evaluate(train_loader)
        val_metrics = self.evaluate(validation_loader)
        end = time.perf_counter()
        inference_time = end - start
        
        
        return TrainingMetrics(
            training_losses=training_losses,
            inference_time=inference_time,
            training_time=train_time,
            training_metrics=train_metrics,
            validation_metrics=val_metrics,
            validation_losses=validation_losses,
        )
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader[T]) -> EvaluationMetrics:
        raise NotImplemented()
    
    def build_loaders(self, dataset: torch.utils.data.Dataset[T], options: TrainingOptions) -> tuple[torch.utils.data.DataLoader[T], torch.utils.data.DataLoader[T]]:
        raise NotImplemented()
    
    def _calculate_metrics(self, all_predictions: np.ndarray, all_targets: np.ndarray, loss: float) -> EvaluationMetrics:
        accuracy = (all_predictions == all_targets).sum() / len(all_predictions)
        f1 = f1_score(all_targets, all_predictions, average="macro")
        precision = precision_score(all_targets, all_predictions, average="macro")
        recall = recall_score(all_targets, all_predictions, average="macro")
        
        return EvaluationMetrics(
            loss=loss,
            accuracy=accuracy,
            f1_macro=f1, # type: ignore
            precision_macro=precision, # type: ignore
            recall_macro=recall, # type: ignore
        )
        
        

class SklearnTrainer(Trainer[T, M]):
    def __init__(self, model):
        self.model = model

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)
    def inference(self, x: np.ndarray) -> bool:
        return np.squeeze(self.model.predict(np.expand_dims(x, axis=0))) == 1

    def build_loaders(self, dataset: torch.utils.data.Dataset[T], options: TrainingOptions) -> tuple[torch.utils.data.DataLoader[T], torch.utils.data.DataLoader[T]]:
        train_dataset, validation_dataset = split_train_val(dataset)
        return torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True), torch.utils.data.DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)

    def train(self, train_loader, val_loader, options: TrainingOptions):
        X, y = next(iter(train_loader))
        self.model.fit(X, y)
        return [], []
    
    def evaluate(self, data_loader):
        all_preds = []
        all_targets = []

        for X_batch, y_batch in tqdm(data_loader):
            # outputs = self.forward(sentence_embedding_model.encode(X_batch))
            outputs = self.forward(X_batch)
            predictions = outputs
            all_preds.append(predictions)
            all_targets.append(y_batch.numpy())
        
        all_preds = np.concat(all_preds)
        all_targets = np.concat(all_targets)

        return self._calculate_metrics(
            all_predictions=all_preds, 
            all_targets=all_targets, 
            loss=0)
        
class TorchTrainer(Trainer[T, M]):
    model: torch.nn.Module
    def __init__(self, model: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

    def build_loaders(self, dataset: torch.utils.data.Dataset[T], options: TrainingOptions) -> tuple[torch.utils.data.DataLoader[T], torch.utils.data.DataLoader[T]]:
        train_dataset, validation_dataset = split_train_val(dataset)
        return (
            torch.utils.data.DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True), 
            torch.utils.data.DataLoader(validation_dataset, batch_size=options.batch_size, shuffle=False)
        )

    def inference(self, x: np.ndarray) -> bool:
        with torch.no_grad():
            result = self.model.forward(torch.as_tensor(x).to(self.device)).cpu()
            return torch.argmax(result).item() == 1
    
    def train(self, train_loader, val_loader, options: TrainingOptions):
        model = self.model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(options.epochs):
            total = 0
            epoch_loss = 0.0
            model.train()
            for (X_batch, y_batch) in tqdm(train_loader):
                X_batch = torch.as_tensor(X_batch)
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
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
                    X_batch = torch.as_tensor(X_batch)
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
                    predictions = torch.argmax(outputs, dim=1)
                    total_correct += (predictions == y_batch).sum().item()
                    total += len(y_batch)
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{options.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {100*total_correct/total:.2f}%")
        
        return train_losses, val_losses

    
    def evaluate(self, data_loader):
        model = self.model.to(self.device)
        
        all_preds = []
        all_targets = []
        
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()

        for X_batch, y_batch in tqdm(data_loader):
            X_batch = torch.as_tensor(X_batch)
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            outputs = model.forward(X_batch)
            total_loss += criterion(outputs, y_batch).item()
            predictions = torch.argmax(outputs, dim=1)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
        
        all_preds = np.concat(all_preds)
        all_targets = np.concat(all_targets)

        return self._calculate_metrics(
            all_predictions=all_preds, 
            all_targets=all_targets, 
            loss=0)
    
    def save(self, filename: str):
        torch.save(self.model, filename)
    
    @classmethod
    def load(cls, filename: str):
        return cls(torch.load(filename))