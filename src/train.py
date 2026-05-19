from dataset_loader import KaggleSpamDataset, TokenizerTransform, ToTensor, EmbeddingDataset, EmbeddingTransform, Word2VecDataset
from models import SimpleRNN, LinearRegression, NeuralNetwork, SimpleLSTM
import torch
from transformers import AutoTokenizer
import time

from trainer import TorchTrainer, SklearnTrainer, Trainer, TrainingOptions, TrainingMetrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from typing import Any

from sentence_transformers import SentenceTransformer

from torch.nn.utils.rnn import pad_sequence

LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-10
EPOCHS = 100
LOG_INTERVAL = 1
BATCH_SIZE = 1

sklearn_classifiers: list[tuple[str, Any, type[Trainer], dict[str, Any]]] = [
    ("SVM", svm.SVC, SklearnTrainer, {}),
    ("Nearest Neighbors (Euclidean)", KNeighborsClassifier, SklearnTrainer, { "metric": "euclidean" }),
    ("Nearest Neighbors (Minkowski)", KNeighborsClassifier, SklearnTrainer, { "metric": "minkowski" }),
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

def test_classifiers(dataset, classifiers: list[tuple[str, Trainer]], suffix: str, metrics, options: TrainingOptions, preprocessing_time: float=0.0):
    for title, classifier in classifiers:
        title += suffix
        print(f"Running {title}")
        out = classifier.run(dataset, options)
        out.training_time += preprocessing_time
        out.inference_time += preprocessing_time
        metrics[title] = out
        
        for k, v in out.model_dump().items():
            if(type(v) == type({})):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2:.4f}")
            elif(type(v) != type([])):
                print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    dataset = KaggleSpamDataset("./data/combined.csv")

    sentence_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    start = time.perf_counter()
    dataset = EmbeddingDataset(dataset,                   
            transform=EmbeddingTransform(sentence_embedding_model)
        )
    end = time.perf_counter()
    preprocessing_time = end - start

    metrics: dict[str, TrainingMetrics] = {}
    test_classifiers(
        dataset=dataset, 
        classifiers=build_classifiers(sklearn_classifiers),
        suffix=" (SentenceTransformer)",
        options=TrainingOptions(
            
        ),
        metrics=metrics,
        preprocessing_time=preprocessing_time)
    test_classifiers(
        dataset=dataset, 
        classifiers=build_classifiers(nn_classifiers, input_size=384),
        suffix=" (SentenceTransformer)",
        metrics=metrics,
        options=TrainingOptions(
            epochs=20,
            learning_rate=2e-5
        ),
        preprocessing_time=preprocessing_time)
    
    dataset = KaggleSpamDataset("./data/combined.csv")

    start = time.perf_counter()
    dataset = Word2VecDataset(dataset)
    end = time.perf_counter()
    preprocessing_time = end - start

    test_classifiers(
        dataset=dataset, 
        classifiers=build_classifiers(sklearn_classifiers),
        suffix=" (Trained Word2Vec)",
        metrics=metrics,
        options=TrainingOptions(
            
        ),
        preprocessing_time=preprocessing_time)
    test_classifiers(
        dataset=dataset, 
        classifiers=build_classifiers(nn_classifiers, input_size=300),
        suffix=" (Trained Word2Vec)",
        metrics=metrics,
        options=TrainingOptions(
            epochs=20,
            learning_rate=2e-5
        ),
        preprocessing_time=preprocessing_time)
        
    dataset = KaggleSpamDataset("./data/combined.csv")

    import gensim.downloader
    model = gensim.downloader.load('glove-wiki-gigaword-300', return_path=False)
    start = time.perf_counter()
    dataset = Word2VecDataset(dataset, model=model)
    end = time.perf_counter()
    preprocessing_time = end - start
    test_classifiers(
        dataset=dataset, 
        classifiers=build_classifiers(sklearn_classifiers),
        suffix=" (Pretrained Word2Vec)",
        metrics=metrics,
        options=TrainingOptions(
        ),
        preprocessing_time=preprocessing_time)
    test_classifiers(
        dataset=dataset, 
        classifiers=build_classifiers(nn_classifiers, input_size=300),
        suffix=" (Pretrained Word2Vec)",
        metrics=metrics,
        options=TrainingOptions(
            epochs=20,
            learning_rate=2e-5
        ),
        preprocessing_time=preprocessing_time)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = KaggleSpamDataset("./data/combined.csv",
        transform=torch.nn.Sequential(
            TokenizerTransform(tokenizer),
            ToTensor()
        ))

    test_classifiers(
        dataset=dataset, 
        classifiers=build_classifiers(tokenizing_classifiers, input_size=30522),
        suffix="",
        metrics=metrics,
        options=TrainingOptions(
            epochs=20,
            learning_rate=1e-6,
            batch_size=16
        ),
        preprocessing_time=preprocessing_time)

    print(metrics)
    
    with open("outputs/results.csv", "w") as f:
        f.write(f"title,val_loss,val_accuracy,val_f1,val_precision,val_recall,train_loss,train_accuracy,train_f1,train_precision,train_recall,train_total_time,train_time_per_sample,inference_total_time,inference_time_per_sample\n")
        
        for k,v in metrics.items():
            f.write(f"\"{k}\",")
            f.write(f"{v.validation_metrics.loss},{v.validation_metrics.accuracy},{v.validation_metrics.f1_macro},{v.validation_metrics.precision_macro},{v.validation_metrics.recall_macro},{v.training_metrics.loss},{v.training_metrics.accuracy},{v.training_metrics.f1_macro},{v.training_metrics.precision_macro},{v.training_metrics.recall_macro},{v.training_time},{v.training_time/len(dataset)},{v.inference_time},{v.inference_time/len(dataset)}")
    
    # with open("outputs/losses.csv", "w") as f:
    #     with_losses = [ v for k,v in metrics.items() if "train_losses" in v]
    #     for v in with_losses:
    #         f.write(f"\"{v["title"]} Train\",")
    #         f.write(f"\"{v["title"]} Val\",")
    #     f.write("\n")
        
    #     for i in range(EPOCHS):
    #         for v in with_losses:
    #             f.write(f"{v["train_losses"][i]},")
    #             f.write(f"{v["val_losses"][i]},")
    #         f.write("\n")
    
    
    # visualize(metrics)
    
    