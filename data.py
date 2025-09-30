import wordnet
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TargetWord:
    def __init__(self, token, wn):
        self.token = token
        self.wn = wn

    def get_candidate_set(self):
        candidates = set()
        candidates.update(self.wn.get_synonyms(self.token))
        candidates.update(set(self.wn.get_hypernyms(self.token)))
        return candidates


class Sentence:
    def __init__(self, sentence, target, value,wn):
        self.tokens = word_tokenize(sentence)
        self.target = TargetWord(target,wn)
        self.value = value
        wnl = WordNetLemmatizer()
        target_index=-1
        if target in self.tokens:
            target_index = self.tokens.index(target)
        else:
            for i, token in enumerate(self.tokens):
                if target == wnl.lemmatize(token.lower(), pos="v"):
                    target_index = i
                    break
        if(target_index<0):
            raise ValueError("Target doesn't appear in sentence")
        else:
            self.context = self.tokens[:target_index] + self.tokens[target_index + 1 :]


class DataSet:

    def __init__(self, filepath, extraction_function):
        self.sentences = extraction_function(filepath)
