from sentence_transformers import SentenceTransformer
from dataset_loader import KaggleDatasets, EmbeddingTransform

class Inferencer:

    def __init__(self):
        print("Loading Sentence Transformer...")
        self.sentence_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.datasets = KaggleDatasets(
            "./data/combined.csv",
            {
                "sentence_transformer_embedding": EmbeddingTransform(self.sentence_embedding_model),
                "word_t_vec": ""
            }
        )



    def fit(self):
        pass
# 