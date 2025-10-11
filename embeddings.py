import fasttext
from sklearn.decomposition import TruncatedSVD
import numpy as np


class FasttextModel:
    def __init__(self, load_file):
        self.model = fasttext.load_model(load_file)
        self.load_file = load_file
        self.output_matrix = self.model.get_output_matrix()

    def get_input_vector(self, token):
        return self.model.get_word_vector(token)

    def get_output_vector(self, token):
        word_id = self.model.get_word_id(token)
        if word_id == -1:
            raise ValueError("Word not in dictionary")
        return self.output_matrix[word_id]

    def get_mean_vector(self, tokens):
        embeddings = [self.get_input_vector(token) for token in tokens]
        return np.mean(embeddings, axis=0)

    # def train(self, min_count=1):
    #     model = fasttext.train_unsupervised(input=self.data_file, model=self.model_type, min_count=min_count)
    #     model.save_model(self.save_file)
    #     self.model = model

    # @staticmethod
    # def loadModel(load_file, model_type, alternative_filename=None):
    #     save_file = load_file if not alternative_filename else alternative_filename
    #     load =
    #     model = FasttextModel(save_file, model_type)
    #     model.model = load
    #     return model


class WordAssociationEmbeddings:
    def __init__(self, swow, index_file, embedding_file):
        self.swow = swow
        self.load(index_file, embedding_file)

    @staticmethod
    def create_graph_embeddings(
        swow, index_file, embedding_file, alpha=0.75, dimensions=300, random_seed=50
    ):
        matrix, indices = swow.get_association_strength_matrix(use_only_cues=True)
        with open(index_file, "w", encoding="utf-8") as output:
            for key in indices:
                output.write(str(key) + "\n")
        embeddings = np.linalg.inv(np.identity(len(indices)) - matrix * alpha)
        svd = TruncatedSVD(n_components=dimensions, random_state=random_seed)
        embeddings = svd.fit_transform(embeddings)
        np.save(embedding_file, embeddings)

    def load(self, index_file, embedding_file):
        self.indices = dict()
        with open(index_file, "r", encoding="utf-8") as input:
            for i, line in enumerate(input):
                self.indices[line.strip()] = i
        self.embeddings = np.load(embedding_file)
