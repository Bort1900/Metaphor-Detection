import fasttext
from sklearn.decomposition import PCA
import numpy as np
from wordnet_interface import WordNetInterface
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from data import Sentence
import time
import os
import torch
from transformers import BertModel, BertTokenizer


class Embeddings:
    def __init__(self):
        """
        Class to create embeddings for tokens
        """
        pass

    def get_mean_vector(self, tokens, use_output_vecs=False):
        """
        returns the mean pooled embeddings for a list of tokens
        tokens: list of tokens whose embeddings are mean pooled
        use_output_vecs: whether to use input or output vectors
        """
        if use_output_vecs:
            embeddings = [self.get_output_vector(token) for token in tokens]
        else:
            embeddings = []
            for token in tokens:
                try:
                    embeddings.append(self.get_input_vector(token))
                except KeyError:
                    continue
        if len(embeddings) == 0:
            raise ValueError("None of the tokens are known")
        return np.mean(embeddings, axis=0)

    def get_input_vector(self, token, pos=None):
        """
        returns the standard embeddings
        :param token: token for which embeddings are returned
        :param pos: part of speech of the token if known
        """
        return token

    def get_output_vector(self, token):
        """
        returns the output embeddings(word2vec) if available
        token: token for which embeddings are returned
        """
        return token


class FasttextModel(Embeddings):
    def __init__(self, load_file):
        """
        Wrapper for Fasttext embeddings
        load_file: filepath where model is stored
        """
        self.model = fasttext.load_model(load_file)
        self.load_file = load_file
        self.output_matrix = self.model.get_output_matrix()
        self.wn_interface = WordNetInterface()

    def get_input_vector(self, token, pos=None):
        """
        returns fasttext word embedding
        :param token: word for which embedding is given
        :param pos: irrelevant
        """
        return self.model.get_word_vector(token)

    def get_output_vector(self, token):
        """
        returns the fasttext output embeddings if available (mean pooling from synonyms,hypernyms if unseen word)
        token: word for which embeddings is given
        """
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
            return self.get_mean_vector(tokens=spare_candidates, use_output_vecs=True)
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
        """
        Class for graph embeddings for Word Association strength graph
        swow: SWOWInterface instance that is used for creating the embeddings
        index_file: where the cue and response indices are loaded from
        embedding_file: where the embedding matrix is loaded from
        """
        self.swow = swow
        self.load(index_file, embedding_file)
        self.mean_vector = np.mean(self.embeddings, axis=0)

    @staticmethod
    def create_graph_embeddings(
        swow,
        index_file,
        embedding_file,
        use_only_cues,
        alpha=0.75,
        dimensions=-1,
    ):
        """
        returns and writes graph embeddings that are created for a given Word association strength graph
        index_file: where the cue and response indices will be stored
        embedding_file: where the embeddings will be stored
        use_only_cues: if True it will calculate the graph embeddings for only the cue data omitting the responses due to long calculation time,
        if False it will give out sparse embeddings from the original strength matrix
        alpha: decay factor representing how much the weight of a connection will diminish with the nodes between them
        dimensions: number of dimensions the embeddings will be projected to, if -1 dimensions will be number of cues/cues + responses
        """
        matrix, indices = swow.get_association_strength_matrix(
            use_only_cues=use_only_cues
        )
        with open(index_file, "w", encoding="utf-8") as output:
            for key in indices:
                output.write(str(key) + "\n")
        if use_only_cues:
            embeddings = np.linalg.inv(np.identity(len(indices)) - matrix * alpha)
        else:
            embeddings = matrix
        if dimensions > 0:
            pca = PCA(n_components=dimensions)
            embeddings = pca.fit_transform(embeddings)
        np.save(embedding_file, embeddings)
        return WordAssociationEmbeddings(
            swow=swow, index_file=index_file, embedding_file=embedding_file
        )

    def load(self, index_file, embedding_file):
        """
        loads the embeddings from save files
        index_file: where the cue and response indices are loaded from
        embedding_file: where the embedding matrix is loaded from
        """
        self.indices = dict()
        with open(index_file, "r", encoding="utf-8") as input:
            for i, line in enumerate(input):
                self.indices[line.strip()] = i
        self.embeddings = np.load(embedding_file)

    def get_input_vector(self, token, pos=None):
        """
        returns embedding for a given cue from the association strength matrix (mean pooling from neighbours if response and only cues are used)
        token: the cue for which embedding is given
        :param pos: irrelevant
        """
        if token in self.indices:
            token_index = self.indices[token]
            return self.embeddings[token_index]
        else:
            neighbouring_nodes = self.swow.get_weighted_neighbours(token)
            total = sum([weight for weight in neighbouring_nodes.values()])
            if len(neighbouring_nodes) == 0 or total == 0:
                raise KeyError(f"{token} is not in word association graph")
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
    def __init__(self, layers):
        """
        Wrapper for Bert Embeddings
        layer: list of which layer(s) of the hidden layers to use as embeddings
        """
        self.layers = layers
        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            output_hidden_states=True,
            cache_dir=os.path.join("/projekte/semrel/WORK-AREA/Users/navid", "bert"),
        )
        self.stops = stopwords.words("english")
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.lookup_table = dict()

    def get_sentence_vector(self, sentence):
        """
        returns contextual Bert embedding for the sentence target
        sentence: Sentence instance for whose target the embedding is given
        """
        tokenized = self.tokenizer(sentence.sentence, return_tensors="pt")
        if torch.cuda.is_available():
            tokenized.to("cuda")
        with torch.no_grad():
            output = self.model(**tokenized)
        if type(sentence.target_index) != int:
            return torch.stack(
                [
                    output.hidden_states[layer][0, i + 1]
                    for layer in self.layers
                    for i in sentence.target_index
                ]
            ).mean(dim=0)
        else:
            return torch.stack(
                [
                    output.hidden_states[layer][0, sentence.target_index + 1]
                    for layer in self.layers
                ]
            ).mean(dim=0)

    def get_mean_vector(self, tokens, use_output_vecs=True):
        """
        returns mean pooled embedding for a list of tokens
        :param tokens: list of tokens whose embeddings are mean pooled
        :param use_output_vecs: irrelevant
        """
        embeddings = []
        for token in tokens:
            embeddings.append(self.get_input_vector(token))
        if None in embeddings:
            print(tokens)
        return torch.stack(embeddings).mean(dim=0)

    def get_input_vector(self, token, pos=None, exclude_sent=None):
        """
        returns the contextual embeddings by mean pooling word net examples for the token

        :param token: token for which the embedding is generated
        :param pos: part of speech the example sentences should be restricted to
        :param exclude_sent: sentence that should not be included in the example sentences
        """
        if (token, pos) in self.lookup_table and not exclude_sent:
            return self.lookup_table[(token, pos)]
        sentences = []
        for synset in wn.synsets(token):
            if not pos or synset.pos() in pos:
                for example in synset.examples():

                    try:
                        sent = Sentence(example, target=token, value=1, pos=pos)
                    except ValueError:
                        continue
                    if not exclude_sent or sent.sentence != exclude_sent.sentence:
                        sentences.append(sent)
        vecs = [self.get_sentence_vector(sentence) for sentence in sentences]
        if len(vecs) == 0:
            if pos == ["n"]:
                sentence = "This is a " + token + "."
            elif pos == ["v"]:
                sentence = "I can " + token + "."
            elif pos == ["a"]:
                sentence = "It's a " + token + " thing."
            else:
                sentence = "What is the meaning of " + token + "?"

            output = self.get_sentence_vector(Sentence(sentence, token, 1, pos=pos))
        else:
            output = torch.stack(vecs).mean(dim=0)

        if not exclude_sent:
            self.lookup_table[(token, pos)] = output
        return output

    def get_context_vector(self, sentence):
        """
        returns mean pooled embedding for sentence context (sentence excluding target)
        sentence: Sentence instance whose embeddings are mean pooled
        """
        embeddings = []
        for i, token in enumerate(sentence.tokens):
            if i != sentence.target_index and token.lower() not in self.stops:
                token_sent = Sentence(
                    sentence=sentence.sentence,
                    target=token,
                    value=sentence.value,
                    phrase=sentence.phrase,
                )
                try:
                    embeddings.append(self.get_sentence_vector(token_sent))
                except KeyError:
                    continue

        return torch.stack(embeddings).mean(dim=0)
