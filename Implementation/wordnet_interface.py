import pandas as pd
import os
from nltk.corpus.reader import wordnet
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


class CandidateSource:
    def __init__(self):
        """
        Class for creating candidate sets for the prediction of best candidates
        """

    def get_candidate_set(self, token: str, pos: list[str] | None = None) -> set:
        """
        returns the candidate set for predicting the best fitting candidate including the token itself
        :param token: the word for which the candidate set is generated:
        :param pos: list of parts of speech to which candidates will be restricted
        """
        return set()


class WordNetInterface(CandidateSource):
    def __init__(self):
        """
        Interface from getting the candidate sets from wordnet synonyms and hypernyms
        """
        self.stops = stopwords.words("english")

    def get_synonyms(self, token: str, pos: str | None = None) -> set:
        """
        gets all the synonymic synsets from wordnet and returns a set of their lemmas

        :param token: the word for which the candidate set is generated
        :param pos: if specified the synonyms will be restricted to this part of speech(v:verb,n:noun,a:adjective)
        """
        synonyms = set()
        for synset in wn.synsets(token):
            if synset and (synset.pos() == pos or not pos):
                lemmas = synset.lemma_names()
                if len(lemmas) == 1 or lemmas[0] != token:
                    synonyms.add(lemmas[0])
                else:
                    synonyms.add(lemmas[1])
        return synonyms

    def get_hypernyms(self, token: str, pos: str | None = None) -> set:
        """
        gets all the hypernymic synsets from wordnet and returns a set of their lemmas

        :param token: the word for which the candidate set is generated
        :param pos: if specified the hypernyms will be restricted to this part of speech(v:verb,n:noun,a:adjective)
        """
        hypernyms = set()
        for synset in wn.synsets(token):
            if synset and synset.hypernyms() and (synset.pos() == pos or not pos):
                for hypernym in synset.hypernyms():
                    lemmas = hypernym.lemma_names()
                    if len(lemmas) == 1 or lemmas[0] != token:
                        hypernyms.add(lemmas[0])
                    else:
                        hypernyms.add(lemmas[1])
        return hypernyms

    def get_candidate_set(self, token: str, pos: list[str] | None = None) -> set:
        """
        gets all the synonymic and hypernymic synsets from wordnet and returns a set of their lemmas and the token

        :param token: the word for which the candidate set is generated
        :param pos: list of parts of speech, if specified the candidates will be restricted to these parts of speech(v:verb,n:noun,a:adjective)
        """
        candidates = set()
        if pos:
            for part in pos:
                candidates.update(self.get_synonyms(token, pos=part))
                candidates.update(set(self.get_hypernyms(token, pos=part)))
        else:
            candidates.update(self.get_synonyms(token))
            candidates.update(set(self.get_hypernyms(token)))
        candidates.difference_update(self.stops)
        candidates.add(token)
        return candidates
