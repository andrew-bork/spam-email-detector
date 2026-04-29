from dataset_loader import KaggleSpamDataset, TokenizerTransform, ToTensor
from rnn import SimpleRNN
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import time

from trainer import TorchTrainer, SklearnTrainer, Trainer
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from typing import Callable

LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-7
EPOCHS = 20
LOG_INTERVAL = 1


classifier_info: list[tuple[str, Callable[..., Trainer], type[Trainer]]] = [
    ("Simple RNN", SimpleRNN, TorchTrainer),
    ("Naive Bayes", GaussianNB, SklearnTrainer),
    ("Nearest Neighbors", NearestNeighbors, SklearnTrainer),
    ("SVM", svm, SklearnTrainer),
]




def build_classifiers():
    return [
        (title, trainer(builder()))
    for title, builder, trainer in classifier_info ]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = KaggleSpamDataset("./data/emails.csv", 
        transform=torch.nn.Sequential(TokenizerTransform(tokenizer), ToTensor())
                                )
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    metrics = {}
    classifiers = build_classifiers()
    for title, classifier in classifiers:
        print(f"\nBuilding loaders...")
        train_loader, val_loader = classifier.build_loaders(train_dataset, val_dataset)

        print(f"\nTraining {title}...")
        start = time.perf_counter()
        classifier.train(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        end = time.perf_counter()
        train_time = end - start

        print(f"\nEvaluating {title}...")
        start = time.perf_counter()
        train_metrics = classifier.evaluate(train_loader)
        val_metrics = classifier.evaluate(val_loader)
        end = time.perf_counter()
        inference_time = end - start
        
        print("\nMetrics:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1_macro']:.4f}, Accuracy: {100*train_metrics['accuracy']:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1_macro']:.4f}, Accuracy: {100*val_metrics['accuracy']:.2f}%")
        metrics[title] = {
            "title": title,
            "validation": val_metrics,
            "train": train_metrics,
            "train_total_time": train_time / n_samples,
            "train_time_per_sample": train_time,
            "inference_total_time": inference_time,
            "inference_time_per_sample": inference_time / n_samples,
        }