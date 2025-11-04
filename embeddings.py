import fasttext
from sklearn.decomposition import PCA
import numpy as np
from wordnet_interface import WordNetInterface
from nltk.corpus import wordnet as wn
from data import Sentence
import time
import os
import torch
from transformers import BertModel, BertTokenizer


class Embeddings:
    def __init__(self):
        pass

    def get_mean_vector(self, tokens, use_input_vecs=True):
        if use_input_vecs:
            embeddings = []
            for token in tokens:
                try:
                    embeddings.append(self.get_input_vector(token))
                except KeyError:
                    continue
        else:
            embeddings = [self.get_output_vector(token) for token in tokens]
        return np.mean(embeddings, axis=0)

    def get_input_vector(self, token):
        return token

    def get_output_vector(self, token):
        return token


class FasttextModel(Embeddings):
    def __init__(self, load_file):
        self.model = fasttext.load_model(load_file)
        self.load_file = load_file
        self.output_matrix = self.model.get_output_matrix()
        self.wn_interface = WordNetInterface(use_pos="")

    def get_input_vector(self, token):
        return self.model.get_word_vector(token)

    def get_output_vector(self, token):
        word_id = self.model.get_word_id(token)
        if word_id == -1:
            spare_candidates = [
                candidate
                for candidate in self.wn_interface.get_candidate_set(token=token)
                if self.model.get_word_id(candidate) >= 0
            ]
            if len(spare_candidates) == 0:
                raise ValueError(
                    f"Could not create output vector for unseen word{token}"
                )
            return self.get_mean_vector(tokens=spare_candidates, use_input_vecs=False)
        return self.output_matrix[word_id]

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


class WordAssociationEmbeddings(Embeddings):
    def __init__(self, swow, index_file, embedding_file):
        self.swow = swow
        self.load(index_file, embedding_file)
        self.mean_vector = np.mean(self.embeddings, axis=0)

    @staticmethod
    def create_graph_embeddings(
        swow,
        index_file,
        embedding_file,
        alpha=0.75,
        dimensions=-1,
    ):
        matrix, indices = swow.get_association_strength_matrix(use_only_cues=True)
        with open(index_file, "w", encoding="utf-8") as output:
            for key in indices:
                output.write(str(key) + "\n")
        embeddings = np.linalg.inv(np.identity(len(indices)) - matrix * alpha)
        if dimensions > 0:
            pca = PCA(n_components=dimensions)
            embeddings = pca.fit_transform(embeddings)
        np.save(embedding_file, embeddings)

    def load(self, index_file, embedding_file):
        self.indices = dict()
        with open(index_file, "r", encoding="utf-8") as input:
            for i, line in enumerate(input):
                self.indices[line.strip()] = i
        self.embeddings = np.load(embedding_file)

    def get_input_vector(self, token):
        if token in self.indices:
            token_index = self.indices[token]
            return self.embeddings[token_index]
        else:
            neighbouring_nodes = self.swow.get_weighted_neighbours(token)
            if len(neighbouring_nodes) == 0:
                raise KeyError(f"{token} is not in word association graph")
            total = sum([weight for weight in neighbouring_nodes.values()])
            map_factor = 1 / total
            weighted_mean = np.zeros(self.mean_vector.shape)
            for neighbour in neighbouring_nodes:
                if neighbour in self.indices:
                    neighbour_index = self.indices[neighbour]
                    weighted_mean += (
                        neighbouring_nodes[neighbour]
                        * self.embeddings[neighbour_index]
                        * map_factor
                    )
            return weighted_mean


class BertEmbeddings(Embeddings):
    def __init__(self, layer):
        self.layer = layer
        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            output_hidden_states=True,
            cache_dir=os.path.join("/projekte/semrel/WORK-AREA/Users/navid", "bert"),
        )

        if torch.cuda.is_available():
            self.model.to("cuda")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def get_input_vector(self, sentence):
        tokenized = self.tokenizer(sentence.sentence, return_tensors="pt")
        if torch.cuda.is_available():
            tokenized.to("cuda")
        with torch.no_grad():
            output = self.model(**tokenized)
        if type(sentence.target_index) != int:
            return torch.stack(
                [
                    output.hidden_states[self.layer][0, i + 1]
                    for i in sentence.target_index
                ]
            ).mean(dim=0)
        else:
            return output.hidden_states[self.layer][0, sentence.target_index + 1]

    def get_mean_vector(self, sentence, use_input_vecs=True):
        embeddings = []
        for token in sentence.tokens:
            token_sent = Sentence(
                sentence=sentence.sentence,
                target=token,
                value=sentence.value,
                phrase=sentence.phrase,
            )
            try:
                embeddings.append(self.get_input_vector(token_sent))
            except KeyError:
                continue
        return torch.stack(embeddings).mean(dim=0)
