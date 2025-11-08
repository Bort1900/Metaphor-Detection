import wordnet_interface
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random
import re
import math
import numpy as np


class Sentence:
    def __init__(self, sentence, target, value, phrase="unknown"):
        """
        A sentence containing a phrase that's either metaphorical or literal
        sentence: string of the sentence
        target: the token or its lemma which is the target for the label
        value: the label, 0: figurative, 1: literal, can also be 0: figurative, 1:unsure, 2:literal
        phrase: the phrase containing the target if known
        """
        self.sentence = sentence
        self.tokens = word_tokenize(sentence)
        self.target = target
        self.value = value
        self.phrase = phrase
        wnl = WordNetLemmatizer()
        self.target_index = -1
        if target in self.tokens:
            self.target_index = self.tokens.index(target)
            self.target_token = target
        else:
            # search for inflection of target
            for i, token in enumerate(self.tokens):
                if target == wnl.lemmatize(token.lower(), pos="v"):
                    self.target_index = i
                    self.target_token = token.lower()
                    break
        if self.target_index < 0:
            raise ValueError("Target doesn't appear in sentence")
        else:
            self.context = (
                self.tokens[: self.target_index] + self.tokens[self.target_index + 1 :]
            )

    def replace_target(self, new_target, split_multi_word=True):
        """
        create a new sentence where the target is replaced by another word
        new_target: the word replacing the old target
        split_multi_word: if flagged, this will look for multi word tokens(indicated by "_") and split them up to replace the old target with multiple words
        """
        if split_multi_word:
            new_sentence = self.sentence.replace(
                self.tokens[self.target_index], " ".join(new_target.split("_")), 1
            )
            target = new_target.split("_")[0]
        else:
            new_sentence = self.sentence.replace(
                self.tokens[self.target_index], new_target, 1
            )
            target = new_target
        output = Sentence(
            sentence=new_sentence,
            target=target,
            value=self.value,
            phrase=self.phrase,
        )
        if split_multi_word:
            output.target_index = [
                output.target_index + i for i in range(len(new_target.split("_")))
            ]
            output.target = new_target.split("_")
            output.target_token = output.target
        return output


class DataSet:
    def __init__(self, filepath, extraction_function, use_unsure):
        """
        A set of sentences
        filepath: where the data that will be converted to Sentences is stored
        extraction_function: function that specifically extracts Sentence instances from the provided datafile must have the use_unsure parameter
        use_unsure: parameter for the extraction_function that specifies wether the Sentence values are binary or have a third "unsure" bin
        """
        self.sentences = extraction_function(filepath, use_unsure)

    def get_splits(self, splits):
        """
        returns train, dev and test splits of the data
        splits: list of 3 values for train, dev and test split proportion in that order
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
        """
        simple cosine similarity function
        """
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
