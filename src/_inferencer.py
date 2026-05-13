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

LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-10
EPOCHS = 100
LOG_INTERVAL = 1
BATCH_SIZE = 1

from typing import Any

sklearn_classifiers: list[tuple[str, Any, type[Trainer], dict[str, Any]]] = [
    # ("Simple RNN", SimpleRNN, TorchTrainer, {"embedding_input_size": 30522}),
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

def build_classifiers(output, x, input_size: int | None = None, suffix:str = ""):
    def build(a):
        title, builder, trainer, kwargs = a
        if(input_size is not None):
            kwargs["input_size"] = input_size
        return (title, trainer(builder(**kwargs)))
    for a in x:
        k,v = build(a)
        output[k + suffix] = v
class Inferencer:

    def __init__(self):
        print("Loading Sentence Transformer...")
        self.sentence_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


        self.data = pd.read_csv("./data/combined.csv")
        corpus = self.data["word_tokenization"] = [tokenize_by_word(t) for t in self.data["text"]]
        self.word_t_vec_trained = Word2Vec(
                    sentences=corpus,
                    vector_size=128,
                    window=5,
                    min_count=1,
                    workers=4,
                    epochs=20,
                    sg=1
                )
        self.word_t_vec_pretrained = gensim.downloader.load('glove-wiki-gigaword-300', return_path=False)
        self.datasets = KaggleDatasets(
            self.data,
            {
                "sentence_transformer_embedding": EmbeddingTransform(self.sentence_embedding_model),
                "word_t_vec_trained": Word2VecTransform(self.word_t_vec_trained),
                "word_t_vec_pretrained": Word2VecTransform(self.word_t_vec_pretrained),
            }
        )

        self.sentence_transformer_classifiers = {}
        build_classifiers(self.sentence_transformer_classifiers, sklearn_classifiers, suffix=" (Sentence Transformer)")
        build_classifiers(self.sentence_transformer_classifiers, nn_classifiers, suffix=" (Sentence Transformer)")

        self.word_t_vec_trained_classifiers = {}
        build_classifiers(self.word_t_vec_trained_classifiers, sklearn_classifiers, suffix=" (Trained Word2Vec)")
        build_classifiers(self.word_t_vec_trained_classifiers, nn_classifiers, suffix=" (Trained Word2Vec)")

        self.word_t_vec_pretrained_classifiers = {}
        build_classifiers(self.word_t_vec_pretrained_classifiers, sklearn_classifiers, suffix=" (Pretrained Word2Vec)")
        build_classifiers(self.word_t_vec_pretrained_classifiers, nn_classifiers, suffix=" (Pretrained Word2Vec)")



    def fit(self):
        train_dataset, val_dataset = split_train_val(self.datasets["sentence_transformer_embedding"])
        for classifier in self.sentence_transformer_classifiers.items():
            train_loader, val_loader = classifier.build_loaders(train_dataset, val_dataset, batch_size=256)

            classifier.train(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    
    # def
    
    def make_inference(self, input: str):
        output = {}

        sentence_transformer_embedding = self.sentence_embedding_model.encode(input).numpy()

        # for k, c in self.sentence_transformer_classifiers.items():
        #     output[k] = c.inference(sentence_transformer_embedding)
        
        output["sentence_transformer_embedding"] = sentence_transformer_embedding
        return output
