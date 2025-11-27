import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


class WordNetInterface:
    def __init__(self):
        # self.work_dir = "/resources/data/WordNet/WordNet-3.0_extract"
        # self.token_to_synset_ids, self.synset_id_to_token, self.hypernym_synsets = (
        #    self.init_index_tables()
        # )
        """
        Interface from getting the candidate sets from wordnet synonyms and hypernyms
        """
        self.stops = stopwords.words("english")

    def init_index_tables(self):
        """
        deprecated
        """
        # token to synset tables
        nouns = pd.read_csv(
            os.path.join(self.work_dir, "noun-synsets.txt"),
            delimiter="\t",
            header=None,
            names=["token", "ids", "ambiguity"],
            dtype={"token": str, "ids": str, "ambiguity": int},
        )
        verbs = pd.read_csv(
            os.path.join(self.work_dir, "verb-synsets.txt"),
            delimiter="\t",
            header=None,
            names=["token", "ids", "ambiguity"],
            dtype={"token": str, "ids": str, "ambiguity": int},
        )
        nouns["POS"] = "NN"
        verbs["POS"] = "V"
        token_to_synset = pd.concat([nouns, verbs])
        token_to_synset["ids"] = token_to_synset["ids"].apply(
            lambda x: [int(id) for id in x.split(",")]
        )

        # synset to token tables
        syn_nouns = pd.read_csv(
            os.path.join(self.work_dir, "synset-nouns.txt"),
            delimiter="\t",
            header=None,
            names=["id", "tokens"],
            dtype={"id": int, "tokens": str},
        )
        syn_verbs = pd.read_csv(
            os.path.join(self.work_dir, "synset-verbs.txt"),
            delimiter="\t",
            header=None,
            names=["id", "tokens"],
            dtype={"id": int, "tokens": str},
        )
        syn_nouns["POS"] = "NN"
        syn_verbs["POS"] = "V"
        synset_to_token = pd.concat([syn_nouns, syn_verbs])
        synset_to_token["tokens"] = synset_to_token["tokens"].apply(
            lambda x: [token.strip() for token in str(x).split(",")]
        )

        # hypernym tables
        noun_hypernyms = pd.read_csv(
            os.path.join(self.work_dir, "hyp_nouns.txt"),
            delimiter="\t",
            header=None,
            names=["hyponym", "hypernyms"],
            dtype={"hyponym": str, "hypernyms": str},
        )
        verb_hypernyms = pd.read_csv(
            os.path.join(self.work_dir, "hyp_verbs.txt"),
            delimiter="\t",
            header=None,
            names=["hyponym", "hypernyms"],
            dtype={"hyponym": str, "hypernyms": str},
        )
        noun_hypernyms["POS"] = "NN"
        verb_hypernyms["POS"] = "V"
        hypernyms = pd.concat([noun_hypernyms, verb_hypernyms])
        hypernyms["hypernyms"] = hypernyms["hypernyms"].apply(
            lambda x: [token.strip() for token in str(x).split(",")]
        )

        return token_to_synset, synset_to_token, hypernyms

    def get_synset_ids(self, token, pos):
        """
        deprecated
        """
        return (
            self.token_to_synset_ids[
                (self.token_to_synset_ids["token"] == token)
                & (self.token_to_synset_ids["POS"] == pos)
            ]["ids"]
            .explode()
            .tolist()
        )

    def get_tokens_from_id(self, id, pos):
        """
        deprecated
        """
        return self.synset_id_to_token[
            (self.synset_id_to_token["id"] == id)
            & (self.synset_id_to_token["POS"] == pos)
        ].iloc[0]["tokens"]

    def get_synonyms(self, token, pos=None):
        """
        gets all the synonymic synsets from wordnet and returns a set of their lemmas

        :param token: the word for which the candidate set is generated
        :param pos: if specified the synonyms will be restricted to this part of speech(v:verb,n:noun,a:adjective)
        """
        synonyms = set()
        synsets = [
            synset for synset in wn.synsets(token) if synset.pos() == pos or not pos
        ]
        for synset in synsets:
            lemmas = synset.lemma_names()
            if len(lemmas) == 1 or lemmas[0] != token:
                synonyms.add(lemmas[0])
            else:
                synonyms.add(lemmas[1])
        return synonyms

    def get_hypernyms(self, token, pos=None):
        """
        gets all the hypernymic synsets from wordnet and returns a set of their lemmas

        :param token: the word for which the candidate set is generated
        :param pos: if specified the hypernyms will be restricted to this part of speech(v:verb,n:noun,a:adjective)
        """
        hypernyms = set()
        synsets = [
            synset.hypernyms()[0]
            for synset in wn.synsets(token)
            if synset.hypernyms() and (synset.pos() == pos or not pos)
        ]
        for synset in synsets:
            lemmas = synset.lemma_names()
            if len(lemmas) == 1 or lemmas[0] != token:
                hypernyms.add(lemmas[0])
            else:
                hypernyms.add(lemmas[1])
        return hypernyms

    def get_candidate_set(self, token, pos=None):
        """
        gets all the synonymic and hypernymic synsets from wordnet and returns a set of their lemmas and the token

        :param token: the word for which the candidate set is generated
        :param pos: if specified the candidates will be restricted to this part of speech(v:verb,n:noun,a:adjective)
        """
        candidates = set()
        candidates.update(self.get_synonyms(token, pos=pos))
        candidates.update(set(self.get_hypernyms(token, pos=pos)))
        candidates.difference_update(self.stops)
        candidates.add(token)
        return candidates
