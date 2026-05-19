from sentence_transformers import SentenceTransformer
from dataset_loader import KaggleDatasets, EmbeddingTransform, tokenize_by_word, Word2VecTransform, split_train_val, PandasColumnDataset

from gensim.models import Word2Vec
import gensim.downloader

import pandas as pd

from trainer import TorchTrainer, SklearnTrainer, Trainer, TrainingOptions
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from models import SimpleRNN, LinearRegression, NeuralNetwork, SimpleLSTM

from function_timer import timeit

from pydantic import BaseModel

from typing import Any
import pathlib
import os
import torch
import numpy as np
from typing import Self

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-10
EPOCHS = 100
LOG_INTERVAL = 1
BATCH_SIZE = 1

knns = [
    ("knn_euclidean", { "metric": "euclidean" }),
    ("knn_minkowski", { "metric": "minkowski" }),
    ("knn_cosine", { "metric": "cosine" }),
]

sklearn_classifiers: list[tuple[str, Any, type[Trainer], dict[str, Any]]] = [
    ("svm", svm.SVC, SklearnTrainer, {}),
    # ("knn_euclidean", KNeighborsClassifier, SklearnTrainer, { "metric": "euclidean" }),
    # ("knn_minkowski", KNeighborsClassifier, SklearnTrainer, { "metric": "minkowski" }),
    # ("knn_cosine", KNeighborsClassifier, SklearnTrainer, { "metric": "cosine" }),
    ("naive_bayes", GaussianNB, SklearnTrainer, {}),
]

nn_classifiers = [
    ("log_reg", LinearRegression, TorchTrainer, { "output_size": 6 }),
    ("nn", NeuralNetwork, TorchTrainer, { "hidden_size": 512, "output_size": 6 }),
]

class SimpleDecision(BaseModel):
    calculation_time: float
    decision: bool

class NearbyNeighbor(BaseModel):
    text: str
    distance: float
    is_spam: bool

class NearestNeighborsDecision(SimpleDecision):
    nearest_neighbors: list[NearbyNeighbor]

class EmbeddingClassifierResults(BaseModel):
    embedding: list[float]
    embedding_calculation_time: float

    svm: SimpleDecision
    # nearest_neighbors_euclidean: SimpleDecision
    # nearest_neighbors_minkowski: SimpleDecision
    # nearest_neighbors_cosine: SimpleDecision
    nearest_neighbors_euclidean: NearestNeighborsDecision
    nearest_neighbors_minkowski: NearestNeighborsDecision
    nearest_neighbors_cosine: NearestNeighborsDecision
    naive_bayes: SimpleDecision
    logistic_regression: SimpleDecision
    neural_network: SimpleDecision

class NonEmbeddingClassifiersResults(BaseModel):
    bert_classifier: SimpleDecision
    
class InferencerResults(BaseModel):
    sentence_transformer: EmbeddingClassifierResults
    word_t_vec_trained: EmbeddingClassifierResults
    word_t_vec_pretrained: EmbeddingClassifierResults
    non_embedding: NonEmbeddingClassifiersResults


def build_classifiers(output, x, input_size: int | None = None, suffix:str = ""):
    def build(a):
        title, builder, trainer, kwargs = a
        if(input_size is not None):
            kwargs["input_size"] = input_size
        return (title, trainer(builder(**kwargs)))
        # return (title, trainer(builder(**kwargs)))
    for a in x:
        k,v = build(a)
        output[k] = v
        
class Inferencer:
    sentence_transformer_classifiers: dict[str, Trainer]
    def __init__(self):
        print("Loading Sentence Transformer...")
        self.sentence_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.data = pd.read_csv("./data/emails.csv")
        corpus = self.data["word_tokenization"] = [tokenize_by_word(t) for t in self.data["text"]]
        self.word_t_vec_trained = Word2Vec(
                    sentences=corpus,
                    vector_size=300,
                    window=5,
                    min_count=1,
                    workers=4,
                    epochs=20,
                    sg=1
                )
        self.word_t_vec_pretrained = gensim.downloader.load('glove-wiki-gigaword-300', return_path=False)
        self.word_t_vec_pretrained.wv = self.word_t_vec_pretrained
        self.datasets = KaggleDatasets(
            self.data,
            {
                "sentence_transformer_embedding": EmbeddingTransform(self.sentence_embedding_model),
                "word_t_vec_trained": Word2VecTransform(self.word_t_vec_trained),
                "word_t_vec_pretrained": Word2VecTransform(self.word_t_vec_pretrained),
            }
        )
        
        self.sentence_transformer_inferencer = EmbeddingInferencer(EmbeddingTransform(self.sentence_embedding_model), self.datasets["sentence_transformer_embedding"], 384)
        self.word_t_vec_trained_inferencer = EmbeddingInferencer(Word2VecTransform(self.word_t_vec_trained), self.datasets["word_t_vec_trained"], 300)
        self.word_t_vec_pretrained_inferencer = EmbeddingInferencer(Word2VecTransform(self.word_t_vec_pretrained), self.datasets["word_t_vec_pretrained"], 300)
        
        self.non_embedding_classifiers = NonEmbeddingInferencer(self.datasets["text"])

    def fit(self):
        # self.sentence_transformer_inferencer.fit()
        return

    def make_inference(self, input: str) -> InferencerResults:
        return InferencerResults(
            sentence_transformer=self.sentence_transformer_inferencer.make_inference(input),
            word_t_vec_trained=self.word_t_vec_trained_inferencer.make_inference(input),
            word_t_vec_pretrained=self.word_t_vec_pretrained_inferencer.make_inference(input),
            non_embedding=self.non_embedding_classifiers.make_inference(input)
        )

    @classmethod
    def load_inferencer(cls, filename: str) -> Self:
        if(pathlib.Path(filename).exists()):
            return torch.load(filename, weights_only=False)
        else:
            a = cls()
            torch.save(a, filename)
            return a

from sklearn.neighbors import NearestNeighbors

class KNNInferencer:
    def __init__(self, metric: str, dataset: PandasColumnDataset):
        self.model = NearestNeighbors(n_neighbors=5, metric=metric)
        self.dataset = dataset
        
        
        X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset))))
        self.model.fit(X, y)
    
    @timeit
    def _get_neighbors(self, input: np.ndarray):
        return self.model.kneighbors(np.asarray([input]))
    
    def make_inference(self, input: np.ndarray) -> NearestNeighborsDecision:
        (distances, neighbors), calculation_time = self._get_neighbors(input)
        distances, neighbors = np.squeeze(distances), np.squeeze(neighbors)
        nearest_neighbors = []
        class_total = 0
        texts, classes = self.dataset.get_raw_sample(neighbors)
        for text, class_index, distance in zip(texts, classes, distances):
            nearest_neighbors.append(NearbyNeighbor(
                text=text,
                distance=distance,
                is_spam=class_index == 1
            ))
            class_total += class_index
        
        
        return NearestNeighborsDecision(
            calculation_time=calculation_time,
            nearest_neighbors=nearest_neighbors,
            decision=2 < class_total
        )
from bert_trainer import BERTTrainer
class NonEmbeddingInferencer:
    def __init__(self, dataset):
        self.bert = BERTTrainer(model_name="bert-base-uncased", max_length=128)
        
        self.bert_options = TrainingOptions(
            batch_size=256,
            epochs=3,
            learning_rate=2e-5,
            weight_decay=0.01
        )
        
        train_loader, val_loader = self.bert.build_loaders(
            dataset, options=self.bert_options
        )
        
        self.bert.train(
            train_loader, val_loader, options=self.bert_options
        )
    
    def make_inference(self, input: str) -> NonEmbeddingClassifiersResults:
        bert_decision, bert_calculation_time = self.bert.timed_inference(input)
        return NonEmbeddingClassifiersResults(
            bert_classifier=SimpleDecision(
                decision=bert_decision,
                calculation_time=bert_calculation_time
            )
        )
        

        

class EmbeddingInferencer:
    def __init__(self, embedder, dataset, input_size):
        self.embedder = embedder
        self.dataset = dataset
        self.classifiers = {}
        build_classifiers(self.classifiers, sklearn_classifiers, suffix=" (Sentence Transformer)")
        build_classifiers(self.classifiers, nn_classifiers, suffix=" (Sentence Transformer)", input_size=input_size)
        
        self.options = TrainingOptions(
            batch_size=256,
            epochs=20,
            learning_rate=2e-5,
            weight_decay=0.01
        )
        
        for classifier in self.classifiers.values():
            train_loader, val_loader = classifier.build_loaders(dataset, options=self.options)
            classifier.train(train_loader, val_loader, options=self.options)

# knns = [
#     ("knn_euclidean", { "metric": "euclidean" }),
#     ("knn_minkowski", { "metric": "minkowski" }),
#     ("knn_cosine", { "metric": "cosine" }),
# ]
        self.knn_classifiers = {
            "knn_euclidean": KNNInferencer("euclidean", self.dataset),
            "knn_minkowski": KNNInferencer("minkowski", self.dataset),
            "knn_cosine": KNNInferencer("cosine", self.dataset),
        }
        
    @timeit
    def _embed(self, input: str):
        return self.embedder(input)

    def _make_simple_embedding_inference(self, embedding, classifier: Trainer) -> SimpleDecision:
        decision, calculation_time = classifier.timed_inference(embedding)
        return SimpleDecision(
            decision=decision,
            calculation_time=calculation_time,
        )
    
    def make_inference(self, input: str) -> EmbeddingClassifierResults:
        embedding, embedding_time  = self._embed(input)
        # embedding = embedding


        svm = self._make_simple_embedding_inference(embedding, self.classifiers["svm"])
        nearest_neighbors_euclidean = self.knn_classifiers["knn_euclidean"].make_inference(embedding)
        nearest_neighbors_minkowski = self.knn_classifiers["knn_minkowski"].make_inference(embedding)
        nearest_neighbors_cosine = self.knn_classifiers["knn_cosine"].make_inference(embedding)
        naive_bayes = self._make_simple_embedding_inference(embedding, self.classifiers["naive_bayes"])
        logistic_regression = self._make_simple_embedding_inference(embedding, self.classifiers["log_reg"])
        neural_network = self._make_simple_embedding_inference(embedding, self.classifiers["nn"])

        return EmbeddingClassifierResults(
            embedding=embedding,
            embedding_calculation_time=embedding_time,
            svm=svm,
            nearest_neighbors_minkowski=nearest_neighbors_minkowski,
            nearest_neighbors_euclidean=nearest_neighbors_euclidean,
            nearest_neighbors_cosine=nearest_neighbors_cosine,
            naive_bayes=naive_bayes,
            logistic_regression=logistic_regression,
            neural_network=neural_network,
        )

    

if __name__ == "__main__":
    inferencer = Inferencer.load_inferencer("./outputs/inferencer.pkl")
    inferencer.fit()
    result = inferencer.make_inference("jadskfljadslkfjdsaklfjdsalfjdsla;jfdskjfdsjakfdsalkfjads")
    print(result)