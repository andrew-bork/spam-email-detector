from dataset_loader import KaggleSpamDataset, TokenizerTransform, ToTensor, EmbeddingDataset, EmbeddingTransform
from models import SimpleRNN, LinearRegression, NeuralNetwork
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import time

from trainer import TorchTrainer, SklearnTrainer, Trainer
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from typing import Callable, Any

from sentence_transformers import SentenceTransformer

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-7
EPOCHS = 50
LOG_INTERVAL = 1


classifier_info: list[tuple[str, Any, type[Trainer], dict[str, Any]]] = [
    # ("Simple RNN", SimpleRNN, TorchTrainer, {"embedding_input_size": 30522}),
    ("3", svm.SVC, SklearnTrainer, {}),
    ("Nearest Neighbors", KNeighborsClassifier, SklearnTrainer, { }),
    ("Naive Bayes", GaussianNB, SklearnTrainer, {}),
    ("Logistic Regression", LinearRegression, TorchTrainer, { "input_size": 384, "output_size": 2 }),
    ("Simple NN", NeuralNetwork, TorchTrainer, { "input_size": 384, "hidden_size": 512 }),
]

tokenizing_classifiers: list[tuple[str, Any, type[Trainer], dict[str, Any]]] = [
    ("Simple RNN", SimpleRNN, TorchTrainer, {"embedding_input_size": 30522}),
]

def build_classifiers():
    return [
        (title, trainer(builder(**kwargs)))
    for title, builder, trainer, kwargs in classifier_info ]

def build_tokenizing_classifiers():
    return [
        (title, trainer(builder(**kwargs)))
    for title, builder, trainer, kwargs in tokenizing_classifiers ]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = KaggleSpamDataset("./data/combined.csv")

    sentence_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    dataset = EmbeddingDataset(dataset,                   
            transform=EmbeddingTransform(sentence_embedding_model)
        )
    
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    metrics = {}
    classifiers = build_classifiers()
    for title, classifier in classifiers:
        print(f"\nBuilding loaders...")
        train_loader, val_loader = classifier.build_loaders(train_dataset, val_dataset, batch_size=64)

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
        print(f"Train time: {train_time} seconds")
        print(f"Inference time: {inference_time} seconds")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1_macro']:.4f}, Accuracy: {100*train_metrics['accuracy']:.2f}%, Precision: {train_metrics['precision_macro']}, Recall: {train_metrics['recall_macro']}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1_macro']:.4f}, Accuracy: {100*val_metrics['accuracy']:.2f}%, Precision: {val_metrics['precision_macro']}, Recall: {val_metrics['recall_macro']}")
        metrics[title] = {
            "title": title,
            "validation": val_metrics,
            "train": train_metrics,
            "train_total_time": train_time / n_samples,
            "train_time_per_sample": train_time,
            "inference_total_time": inference_time,
            "inference_time_per_sample": inference_time / n_samples,
        }
    

    dataset = KaggleSpamDataset("./data/combined.csv",
        transform=torch.nn.Sequential(
            TokenizerTransform(tokenizer),
            ToTensor()
        ))
    
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    tokenizing_classifiers = build_tokenizing_classifiers()
    for title, classifier in tokenizing_classifiers:
        print(f"\nBuilding loaders...")
        train_loader, val_loader = classifier.build_loaders(train_dataset, val_dataset, batch_size=1)

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
        print(f"Train time: {train_time} seconds")
        print(f"Inference time: {inference_time} seconds")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1_macro']:.4f}, Accuracy: {100*train_metrics['accuracy']:.2f}%, Precision: {train_metrics['precision_macro']}, Recall: {train_metrics['recall_macro']}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1_macro']:.4f}, Accuracy: {100*val_metrics['accuracy']:.2f}%, Precision: {val_metrics['precision_macro']}, Recall: {val_metrics['recall_macro']}")
        metrics[title] = {
            "title": title,
            "validation": val_metrics,
            "train": train_metrics,
            "train_total_time": train_time / n_samples,
            "train_time_per_sample": train_time,
            "inference_total_time": inference_time,
            "inference_time_per_sample": inference_time / n_samples,
        }



    print(metrics)