import wordnet_interface
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random
import math
import numpy as np


class Sentence:
    def __init__(self, sentence, target, value, phrase="unknown"):
        """
        sentence: string of the sentence
        target: the token or its lemma which is the target for the label
        value: the label, 0: literal, 1: figurative
        phrase: the phrase containing the target if known
        """
        self.tokens = word_tokenize(sentence)
        self.target = target
        self.value = value
        self.phrase = phrase
        wnl = WordNetLemmatizer()
        target_index = -1
        if target in self.tokens:
            target_index = self.tokens.index(target)
        else:
            for i, token in enumerate(self.tokens):
                if target == wnl.lemmatize(token.lower(), pos="v"):
                    target_index = i
                    break
        if target_index < 0:
            raise ValueError("Target doesn't appear in sentence")
        else:
            self.context = self.tokens[:target_index] + self.tokens[target_index + 1 :]


class DataSet:

    def __init__(self, filepath, extraction_function):
        self.sentences = extraction_function(filepath)

    def get_splits(self, splits):
        """
        returns train, dev and test splits of the data, proportions can be given in this order
        """
        if len(splits) != 3:
            raise ValueError(
                "Please give one value for train, development and test split each"
            )
        if sum(splits) > 1:
            raise ValueError("Splits mustn't add to more than 100%")
        num_train = math.floor(len(self.sentences) * splits[0])
        num_dev = math.floor(len(self.sentences) * splits[1])
        num_test = math.floor(len(self.sentences) * splits[2])
        partitions = random.sample(self.sentences, k=num_train + num_dev + num_test)
        return (
            partitions[:num_train],
            partitions[num_train : num_train + num_dev],
            partitions[num_train + num_dev : num_train + num_dev + num_test],
        )


class Vectors:
    @staticmethod
    def cos_sim(vec1, vec2):
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
