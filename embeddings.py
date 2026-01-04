from typing import Self, overload
import fasttext
from sklearn.decomposition import PCA
import numpy as np
from torch import Value
from wordnet_interface import WordNetInterface
from swow_interface import SWOWInterface
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from data import Sentence
import time
import os
from tqdm import tqdm
import torch
import random
from transformers import BertModel, BertTokenizerFast
from gensim.models import Word2Vec, KeyedVectors


class Embeddings:
    def __init__(self):
        """
        Class to create embeddings for tokens
        """
        pass

    def get_mean_vector(
        self, tokens: list[str], use_output_vecs: bool = False
    ) -> np.ndarray:
        """
        returns the mean pooled embeddings for a list of tokens
        tokens: list of tokens whose embeddings are mean pooled
        use_output_vecs: whether to use input or output vectors
        """
        if use_output_vecs:
            for token in tokens:
                try:
                    embeddings = [self.get_output_vector(token) for token in tokens]
                except ValueError:
                    continue
        else:
            embeddings = []
            for token in tokens:
                try:
                    embeddings.append(self.get_input_vector(token))
                except KeyError:
                    continue
        if len(embeddings) == 0:
            raise ValueError("None of the tokens are known")
        return np.mean(a=embeddings, axis=0)

    def get_input_vector(
        self, token: str, exclude_sent: Sentence | None = None, pos: str | None = None
    ) -> np.ndarray:
        """
        returns the standard embeddings
        :param token: token for which embeddings are returned
        :param pos: part of speech of the token if known
        """
        return np.zeros([1])

    def get_output_vector(self, token: str) -> np.ndarray:
        """
        returns the output embedding for the token(word2vec)

        :param token: token for which embedding is returned
        """
        return np.zeros([1])


class FasttextModel(Embeddings):
    def __init__(self, load_file: str):
        """
        Wrapper for Fasttext embeddings
        load_file: filepath where model is stored
        """
        self.model = fasttext.load_model(load_file)
        self.load_file = load_file
        self.output_matrix = self.model.get_output_matrix()
        self.wn_interface = WordNetInterface()

    def get_input_vector(
        self,
        token: str,
        exclude_sent: Sentence | None = None,
        pos: str | None = None,
    ) -> np.ndarray:
        """
        returns fasttext word embedding
        :param token: word for which embedding is given
        :param pos: irrelevant
        :param exclude_sent: irrelevant
        """
        return self.model.get_word_vector(token)

    def get_output_vector(self, token: str) -> np.ndarray:
        """
        returns the fasttext output embeddings if available (mean pooling from synonyms,hypernyms if unseen word)
        token: word for which embeddings is given
        """
        word_id = self.model.get_word_id(token)
        if word_id == -1:
            pool = [token]
            spare_candidates = []
            while len(spare_candidates) == 0:
                pool_size = len(pool)
                pool += [
                    candidate
                    for pool_candidate in pool
                    for candidate in self.wn_interface.get_candidate_set(
                        token=pool_candidate
                    )
                ]
                spare_candidates += [
                    candidate
                    for candidate in pool
                    if self.model.get_word_id(candidate) >= 0
                ]
                if len(spare_candidates) == 0 and len(pool) == pool_size:
                    raise ValueError("Could not create embedding for unknowwn word")
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
    def __init__(self, swow: SWOWInterface, index_file: str, embedding_file: str):
        """
        Class for graph embeddings for Word Association strength graph
        swow: SWOWInterface instance that is used for creating the embeddings
        index_file: where the cue and response indices are loaded from
        embedding_file: where the embedding matrix is loaded from
        """
        self.swow = swow
        self.load(index_file, embedding_file)

    @classmethod
    def create_graph_embeddings(
        cls,
        swow: SWOWInterface,
        index_file: str,
        embedding_file: str,
        use_only_cues: bool,
        alpha: float = 0.75,
        dimensions: int = -1,
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

    def load(self, index_file: str, embedding_file: str):
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

    def get_input_vector(
        self, token: str, exclude_sent: Sentence | None = None, pos: str | None = None
    ) -> np.ndarray:
        """
        returns embedding for a given cue from the association strength matrix (mean pooling from neighbours if response and only cues are used)
        token: the cue for which embedding is given
        :param pos: irrelevant
        :param exclude_sent: irrelevant
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
            weighted_mean = np.zeros_like(self.embeddings[0])
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
    def __init__(
        self,
        layers: list[int],
        use_phrase_embedding: str | None = None,
        mean_weights: list[float] = [],
    ):
        """
        Wrapper for Bert Embeddings
        :param layer: list of which layer(s) of the hidden layers to use as embeddings
        :param use_phrase_embedding: whether to generate embedding for whole phrase or just target word, also specifies the meaning method: "mean","weighted","max","concat", if "weighted" needs to specify weights as "mean_weights"
        :param mean_weights: weights for verb and object embeddings for weighted mean when using "weighted" as phrase_embedding

        """
        self.layers = layers
        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            output_hidden_states=True,
            cache_dir=os.path.join("/projekte/semrel/WORK-AREA/Users/navid", "bert"),
        )
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.lookup_table = dict()
        self.use_phrase_embedding = use_phrase_embedding
        self.mean_weights = mean_weights

    def get_sentence_vector(self, sentence):
        """
        returns contextual Bert embedding for the sentence target
        sentence: Sentence instance for whose target the embedding is given
        """
        if self.use_phrase_embedding and sentence.phrase != "unknown":
            return self.get_phrase_embedding(sentence)
        tokenized = self.tokenizer(
            sentence.tokens, return_tensors="pt", is_split_into_words=True
        )
        word_ids = tokenized.word_ids()
        if torch.cuda.is_available():
            tokenized.to("cuda")
        with torch.no_grad():
            output = self.model(**tokenized)
        if type(sentence.target_index) != int:
            subtoken_indices = [
                subtoken_index
                for subtoken_index, token_index in enumerate(word_ids)
                if token_index in sentence.target_index
            ]
            return torch.stack(
                [
                    output.hidden_states[layer][0, i]
                    for layer in self.layers
                    for i in subtoken_indices
                ]
            ).mean(dim=0)
        else:
            subtoken_indices = [
                subtoken_index
                for subtoken_index, token_index in enumerate(word_ids)
                if token_index == sentence.target_index
            ]
            return torch.stack(
                [
                    output.hidden_states[layer][0, i]
                    for layer in self.layers
                    for i in subtoken_indices
                ]
            ).mean(dim=0)

    def get_phrase_embedding(self, sentence):
        """
        returns the embedding of the sentence phrase according to the specified mean method

        :param sentence: sentence that will be predicted
        """
        if sentence.phrase == "unknown":
            return self.get_sentence_vector(sentence)
        phrase = sentence.phrase.split()
        verb_sentence = sentence.change_target(new_target=phrase[0], new_pos="v")
        noun_sentence = sentence.change_target(new_target=phrase[1], new_pos="n")
        verb_embedding = self.get_sentence_vector(verb_sentence)
        noun_embedding = self.get_sentence_vector(noun_sentence)
        if self.use_phrase_embedding == "mean":
            phrase_embedding = torch.stack([verb_embedding, noun_embedding]).mean(dim=0)
        elif self.use_phrase_embedding == "weighted":
            if len(self.mean_weights) != 2:
                raise ValueError(
                    "must specify weights for verb and object as mean_weights"
                )
            phrase_embedding = torch.stack(
                [
                    verb_embedding * self.mean_weights[0],
                    noun_embedding * self.mean_weights[1],
                ]
            ).sum(dim=0) / sum(self.mean_weights)
        elif self.use_phrase_embedding == "max":
            stacked = torch.stack([verb_embedding, noun_embedding])
            abs_values, indices = stacked.abs().max(dim=0)
            phrase_embedding = stacked.gather(0, indices.unsqueeze(0)).squeeze(0)
        elif self.use_phrase_embedding == "concat":
            phrase_embedding = torch.concat((verb_embedding, noun_embedding))
        else:
            raise ValueError(
                "use_phrase_embedding must be one of 'mean', 'weighted', 'max', 'concat'"
            )
        return phrase_embedding

    def get_mean_vector(
        self, tokens: list[str], use_output_vecs: bool = True
    ) -> torch.Tensor:
        """
        returns mean pooled embedding for a list of tokens
        :param tokens: list of tokens whose embeddings are mean pooled
        :param use_output_vecs: irrelevant
        """
        embeddings = []
        for token in tokens:
            try:
                embeddings.append(self.get_input_vector(token))
            except ValueError:
                continue
        if None in embeddings:
            print(tokens)
        return torch.stack(embeddings).mean(dim=0)

    def get_input_vector(
        self, token: str, pos: str | None = None, exclude_sent=None
    ) -> torch.Tensor:
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
            if synset and (not pos or synset.pos() == pos):
                for example in synset.examples():
                    try:
                        sent = Sentence(example, target=token, value=1, pos=pos)
                    except ValueError:
                        continue
                    if (
                        not exclude_sent
                        or sent.sentence != exclude_sent.sentence.strip()
                    ):
                        sentences.append(sent)
                    if (
                        exclude_sent
                        and exclude_sent.sentence.strip() == sent.sentence.strip()
                    ):
                        breakpoint()
        vecs = [self.get_sentence_vector(sentence) for sentence in sentences]
        if len(vecs) == 0:
            if pos == "n":
                sentence = "This is a " + token + "."
            elif pos == "v":
                sentence = "I can " + token + "."
            elif pos == "a":
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
            if i != sentence.target_index:
                token_sent = sentence.change_target(
                    new_target=token, new_target_index=i
                )
                try:
                    embeddings.append(self.get_sentence_vector(token_sent))
                except KeyError:
                    continue
        # sentence context might only contain stopwords
        if len(embeddings) == 0:
            return self.get_sentence_vector(sentence)
        return torch.stack(embeddings).mean(dim=0)


class Node2VecEmbeddings(Embeddings):
    def __init__(self, loadfile: str, swow: SWOWInterface):
        """
        class for loading and returning node2vec embeddings

        :param loadfile: The path to where the keyedvectors of the model are stored
        :param swow: SWOWInterface used to retrieve neighbouring nodes
        """
        self.wordvectors = KeyedVectors.load(loadfile)
        self.swow = swow

    def get_input_vector(
        self,
        token: str,
        exclude_sent: Sentence | None = None,
        pos: str | None = None,
    ):
        """
        returns the standard embeddings
        :param token: token for which embeddings are returned
        :param pos: irrelevant
        :param exclude_sent: irrelevant
        """
        if token in self.wordvectors:
            return self.wordvectors[token]

        else:
            neighbouring_nodes = self.swow.get_weighted_neighbours(token)
            total = sum([weight for weight in neighbouring_nodes.values()])
            if len(neighbouring_nodes) == 0 or total == 0:
                raise KeyError(f"{token} is not in word association graph")
            map_factor = 1 / total
            weighted_mean = np.zeros(self.wordvectors.vector_size)
            for neighbour in neighbouring_nodes:
                if neighbour in self.wordvectors:
                    weighted_mean += (
                        neighbouring_nodes[neighbour]
                        * self.wordvectors[neighbour]
                        * map_factor
                    )
            return weighted_mean


class Node2VecEmbeddingsCreator:
    def __init__(self, graph: SWOWInterface, is_directed=False, p=1, q=1):
        """
        Class for creating Node2Vec Embeddings following the original implementation(slightly adapted) by Aditya Grover and Jure Leskovec.
        See https://github.com/aditya-grover/node2vec/tree/master

        :param graph: the graph as an instance of SWOWInterface
        :param is_directed: whether the graph should be seen as directed (cues to responses)
        :param p: return parameter, controls the likelihood of returning to the previous node during the random walk(smaller value-> more local walks)
        :param q: in-out parameter, controls the likelihood of further graph exploration (smaller value-> more depth first behaviour, higher value-> more breadth first behaviour)
        """

        self.graph = graph
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.nodes = self.graph.get_nodes(only_cues=self.is_directed)
        self.alias_nodes = dict()
        self.alias_edges = dict()

    def create_embeddings(
        self,
        save_file: str,
        walks: list[list[str]],
        dimensions: int,
        window_size: int,
        min_count=0,
        sg=False,
    ):
        """
        Creates and saves

        :param save_file: where the word2vec model will be stored, word2vec format without file ending
        :param walks: the random walks to train the word2vec model
        :param dimensions: number of dimensions the embeddings will have
        :param window_size: window around the node in the walk that will be taken for positive sampling
        :param sg: whether the model should learn with Skip-gram or CBOW
        """
        model = Word2Vec(
            sentences=walks,
            vector_size=dimensions,
            window=window_size,
            sg=sg,
        )
        model.save(save_file + ".model")
        wv = model.wv
        wv.save(save_file + ".kv")

    def node2vec_walk(self, walk_length: int, start_node: str) -> list[str]:
        """
        Simulate a random walk starting from start node.

        :param walk_length: length of a random walk
        :start_node:  the node to start from
        """

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(
                self.graph.get_neighbours(cur, directional=self.is_directed)
            )
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    alias_node = self.get_alias_nodes(cur)
                    walk.append(cur_nbrs[self.alias_draw(alias_node[0], alias_node[1])])
                else:
                    prev = walk[-2]
                    alias_edge = self.get_alias_edge(prev, cur)
                    next = cur_nbrs[
                        self.alias_draw(
                            alias_edge[0],
                            alias_edge[1],
                        )
                    ]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks: int, walk_length: int) -> list[list[str]]:
        """
        Repeatedly simulate random walks from each node.

        :param num_walks: number of random walks performed from each node as starting point
        :param walk_length: the length of the random walks
        """
        walks = []
        nodes = [node for node in self.nodes]
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), "/", str(num_walks))
            random.shuffle(nodes)
            for node in tqdm(nodes):
                walks.append(
                    self.node2vec_walk(walk_length=walk_length, start_node=node)
                )
        return walks

    def get_alias_edge(self, src: str, dst: str) -> tuple[list[int], list[float]]:
        """
        Get the alias edge setup lists for a given edge, i.e. aliased probabilities for getting to the neighbours after going from src to dst
        """
        if (src, dst) in self.alias_edges:
            return self.alias_edges[(src, dst)]
        unnormalized_probs = []
        total = 0
        for dst_nbr in sorted(
            self.graph.get_neighbours(dst, directional=self.is_directed)
        ):
            if dst_nbr == src:
                strength = (
                    self.graph.get_association_strength(
                        dst, dst_nbr, directional=self.is_directed
                    )
                    / self.p
                )
                unnormalized_probs.append(strength)
                total += strength
            elif dst_nbr in self.graph.get_neighbours(
                src, directional=self.is_directed
            ):
                strength = self.graph.get_association_strength(
                    dst, dst_nbr, directional=self.is_directed
                )
                unnormalized_probs.append(strength)
                total += strength
            else:
                strength = (
                    self.graph.get_association_strength(
                        dst, dst_nbr, directional=self.is_directed
                    )
                    / self.q
                )
                unnormalized_probs.append(strength)
                total += strength
        normalized_probs = [float(u_prob) / total for u_prob in unnormalized_probs]
        self.alias_edges[(src, dst)] = self.alias_setup(normalized_probs)
        return self.alias_edges[(src, dst)]

    def get_alias_nodes(self, node: str):
        """
        get the aliased transition_probabilities for a node
        """
        if node in self.alias_nodes:
            return self.alias_nodes[node]
        unnormalized_probs = []
        total = 0
        for nbr in sorted(
            self.graph.get_neighbours(node, directional=self.is_directed)
        ):
            strength = self.graph.get_association_strength(
                node, nbr, directional=self.is_directed
            )
            unnormalized_probs.append(strength)
            total += strength
        normalized_probs = [float(u_prob) / total for u_prob in unnormalized_probs]
        self.alias_nodes[node] = self.alias_setup(normalized_probs)
        return self.alias_nodes[node]

    @staticmethod
    def alias_setup(probs):
        """
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    @staticmethod
    def alias_draw(J, q):
        """
        Draw sample from a non-uniform discrete distribution using alias sampling.
        """
        K = len(J)

        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
