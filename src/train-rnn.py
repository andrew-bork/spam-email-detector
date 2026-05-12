from dataset_loader import KaggleSpamDataset, TokenizerTransform, ToTensor, EmbeddingDataset, EmbeddingTransform, Word2VecDataset
from models import SimpleRNN, LinearRegression, NeuralNetwork, SimpleLSTM
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

from torch.nn.utils.rnn import pad_sequence


LEARNING_RATE = 1e-8
WEIGHT_DECAY = 1e-9
EPOCHS = 100
LOG_INTERVAL = 1
BATCH_SIZE = 1

sklearn_classifiers: list[tuple[str, Any, type[Trainer], dict[str, Any]]] = [
    # ("Simple RNN", SimpleRNN, TorchTrainer, {"embedding_input_size": 30522}),
    ("SVM", svm.SVC, SklearnTrainer, {}),
    ("Nearest Neighbors (Euclidean)", KNeighborsClassifier, SklearnTrainer, { }),
    ("Nearest Neighbors (Cosine)", KNeighborsClassifier, SklearnTrainer, { "metric": "cosine" }),
    ("Naive Bayes", GaussianNB, SklearnTrainer, {}),
]

nn_classifiers = [
    ("Logistic Regression", LinearRegression, TorchTrainer, { "output_size": 6 }),
    ("Simple NN", NeuralNetwork, TorchTrainer, { "hidden_size": 512, "output_size": 6 }),
]

tokenizing_classifiers: list[tuple[str, Any, type[Trainer], dict[str, Any]]] = [
    ("Simple RNN", SimpleRNN, TorchTrainer, { "output_size": 2 }),
    ("Simple LSTM", SimpleLSTM, TorchTrainer, { "output_size": 2 }),
]

def pad_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """
    Collate variable-length token sequences into a padded batch.
 
    Each item in `batch` is (token_ids: 1-D LongTensor, label: 0-D LongTensor).
    Returns:
        padded : (batch_size, max_seq_len)  — zero-padded (padding_idx=0 in Embedding)
        labels : (batch_size,)
    """
    sequences, labels = zip(*batch)
    padded  = pad_sequence(sequences, batch_first=True, padding_value=0)  # (B, T)
    labels  = torch.stack(labels)                                          # (B,)
    return padded, labels



def build_classifiers(x, input_size: int | None = None):
    def build(a):
        title, builder, trainer, kwargs = a
        if(input_size is not None):
            kwargs["input_size"] = input_size
        return (title, trainer(builder(**kwargs)))
    return [
        build(a)
    for a in x ]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = KaggleSpamDataset("./data/emails.csv",
        transform=torch.nn.Sequential(
            TokenizerTransform(tokenizer),
            ToTensor()
        ))
    
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    model = SimpleLSTM(30522)
    trainer = TorchTrainer(model)
    
    metrics = {}
    title = "Simple RNN"
    
    print(f"\nBuilding loaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )


    print(f"\nTraining {title}...")
    start = time.perf_counter()
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=EPOCHS, lr=1e-7, weight_decay=WEIGHT_DECAY)
    end = time.perf_counter()
    train_time = end - start

    print(f"\nEvaluating {title}...")
    start = time.perf_counter()
    train_metrics = trainer.evaluate(train_loader)
    val_metrics = trainer.evaluate(val_loader)
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
        "train_total_time": train_time,
        "train_time_per_sample": train_time / n_samples,
        "inference_total_time": inference_time,
        "inference_time_per_sample": inference_time / n_samples,
        "train_losses": train_losses,
        "val_losses": val_losses,
        # "parameter_count": sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
    }


    


    print(metrics)
    
    
    with open("outputs/results-lstm.csv", "w") as f:
        f.write(f"title,val_loss,val_accuracy,val_f1,val_precision,val_recall,train_loss,train_accuracy,train_f1,train_precision,train_recall,train_total_time,train_time_per_sample,inference_total_time,inference_time_per_sample\n")
        for k,v in metrics.items():
            f.write(f"\"{k}\",")
            for k2, v2 in v["validation"].items():
                f.write(f"{v2},")
            for k2, v2 in v["train"].items():
                f.write(f"{v2},")
            f.write(f"{v["train_total_time"]},{v["train_time_per_sample"]},{v["inference_total_time"]},{v["inference_time_per_sample"]}\n")
    
            
    
    with open("outputs/losses-lstm.csv", "w") as f:
        with_losses = [ v for k,v in metrics.items() if "train_losses" in v]
        for v in with_losses:
            f.write(f"\"{v["title"]} Train\",")
            f.write(f"\"{v["title"]} Val\",")
        f.write("\n")
        
        for i in range(EPOCHS):
            for v in with_losses:
                f.write(f"{v["train_losses"][i]},")
                f.write(f"{v["val_losses"][i]},")
            f.write("\n")
    
    
    # visualize(metrics)
    
    