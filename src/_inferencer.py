from sentence_transformers import SentenceTransformer
from dataset_loader import KaggleDatasets, EmbeddingTransform, tokenize_by_word, Word2VecTransform, split_train_val

from gensim.models import Word2Vec
import gensim.downloader

import pandas as pd

from trainer import TorchTrainer, SklearnTrainer, Trainer
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from models import SimpleRNN, LinearRegression, NeuralNetwork, SimpleLSTM

from function_timer import timeit

from pydantic import BaseModel

from typing import Any

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-10
EPOCHS = 100
LOG_INTERVAL = 1
BATCH_SIZE = 1

sklearn_classifiers: list[tuple[str, Any, type[Trainer], dict[str, Any]]] = [
    ("svm", svm.SVC, SklearnTrainer, {}),
    ("knn_euclidean", KNeighborsClassifier, SklearnTrainer, { "metric": "euclidean" }),
    ("knn_minkowski", KNeighborsClassifier, SklearnTrainer, { "metric": "minkowski" }),
    ("knn_cosine", KNeighborsClassifier, SklearnTrainer, { "metric": "cosine" }),
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

class NearestNeighborsDecision(SimpleDecision):
    nearest_neighbors: list[NearbyNeighbor]

class EmbeddingClassifierResults(BaseModel):
    embedding: list[float]
    embedding_calculation_time: float

    svm: SimpleDecision
    nearest_neighbors_euclidean: SimpleDecision
    nearest_neighbors_minkowski: SimpleDecision
    nearest_neighbors_cosine: SimpleDecision
    # nearest_neighbors_euclidean: NearestNeighborsDecision
    # nearest_neighbors_minkowski: NearestNeighborsDecision
    # nearest_neighbors_cosine: NearestNeighborsDecision
    naive_bayes: SimpleDecision
    logistic_regression: SimpleDecision
    neural_network: SimpleDecision
    
class InferencerResults(BaseModel):
    sentence_transformer: EmbeddingClassifierResults


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
        # corpus = self.data["word_tokenization"] = [tokenize_by_word(t) for t in self.data["text"]]
        # self.word_t_vec_trained = Word2Vec(
        #             sentences=corpus,
        #             vector_size=128,
        #             window=5,
        #             min_count=1,
        #             workers=4,
        #             epochs=20,
        #             sg=1
        #         )
        # self.word_t_vec_pretrained = gensim.downloader.load('glove-wiki-gigaword-300', return_path=False)
        self.datasets = KaggleDatasets(
            self.data,
            {
                "sentence_transformer_embedding": EmbeddingTransform(self.sentence_embedding_model),
                # "word_t_vec_trained": Word2VecTransform(self.word_t_vec_trained),
                # "word_t_vec_pretrained": Word2VecTransform(self.word_t_vec_pretrained),
            }
        )

        self.sentence_transformer_classifiers = {}
        build_classifiers(self.sentence_transformer_classifiers, sklearn_classifiers, suffix=" (Sentence Transformer)")
        build_classifiers(self.sentence_transformer_classifiers, nn_classifiers, suffix=" (Sentence Transformer)", input_size=384)

        # self.word_t_vec_trained_classifiers = {}
        # build_classifiers(self.word_t_vec_trained_classifiers, sklearn_classifiers, suffix=" (Trained Word2Vec)")
        # build_classifiers(self.word_t_vec_trained_classifiers, nn_classifiers, suffix=" (Trained Word2Vec)")

        # self.word_t_vec_pretrained_classifiers = {}
        # build_classifiers(self.word_t_vec_pretrained_classifiers, sklearn_classifiers, suffix=" (Pretrained Word2Vec)")
        # build_classifiers(self.word_t_vec_pretrained_classifiers, nn_classifiers, suffix=" (Pretrained Word2Vec)")



    def fit(self):
        train_dataset, val_dataset = split_train_val(self.datasets["sentence_transformer_embedding"])
        for classifier in self.sentence_transformer_classifiers.values():
            train_loader, val_loader = classifier.build_loaders(train_dataset, val_dataset, batch_size=256)

            classifier.train(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    

    @timeit
    def _sentence_embedding_encode(self, input: str):
        return self.sentence_embedding_model.encode(input)

    def _make_simple_embedding_inference(self, embedding, classifier: Trainer) -> SimpleDecision:
        decision, calculation_time = classifier.timed_inference(embedding)
        return SimpleDecision(
            decision=decision,
            calculation_time=calculation_time,
        )

    def _make_embedding_inference(self, input: str, embedder, classifiers):

        embedding, embedding_time  = embedder(input)
        # embedding = embedding


        svm = self._make_simple_embedding_inference(embedding, classifiers["svm"])
        nearest_neighbors_euclidean = self._make_simple_embedding_inference(embedding, classifiers["knn_euclidean"])
        nearest_neighbors_minkowski = self._make_simple_embedding_inference(embedding, classifiers["knn_minkowski"])
        nearest_neighbors_cosine = self._make_simple_embedding_inference(embedding, classifiers["knn_cosine"])
        naive_bayes = self._make_simple_embedding_inference(embedding, classifiers["naive_bayes"])
        logistic_regression = self._make_simple_embedding_inference(embedding, classifiers["log_reg"])
        neural_network = self._make_simple_embedding_inference(embedding, classifiers["nn"])

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

    def make_inference(self, input: str) -> InferencerResults:
        return InferencerResults(
            sentence_transformer=self._make_embedding_inference(input, self._sentence_embedding_encode, self.sentence_transformer_classifiers)
        )




if __name__ == "__main__":
    inferencer = Inferencer()
    inferencer.fit()
    result = inferencer.make_inference("jadskfljadslkfjdsaklfjdsalfjdsla;jfdskjfdsjakfdsalkfjads")
    print(result)