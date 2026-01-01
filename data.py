import wordnet_interface
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random
import re
import math
import numpy as np


class Sentence:
    def __init__(self, sentence, target, value, phrase="unknown", pos=None):
        """
        A sentence containing a phrase that's either metaphorical or literal
        sentence: string of the sentence
        :param target: the token or its lemma which is the target for the label
        :param value: the label, 0: figurative, 1: literal, can also be 0: figurative, 1:unsure, 2:literal
        :param phrase: the phrase containing the target if known
        :param pos: the part of speech of the target word if known
        """
        self.sentence = sentence
        self.tokens = word_tokenize(sentence)
        self.target = target
        self.value = value
        self.phrase = phrase
        self.pos = pos
        wnl = WordNetLemmatizer()
        self.target_index = -1
        if target in self.tokens:
            self.target_index = self.tokens.index(target)
            self.target_token = target
        else:
            # search for inflection of target
            for i, token in enumerate(self.tokens):
                if target == wnl.lemmatize(token.lower(), pos="n" if not pos else pos):
                    self.target_index = i
                    self.target_token = token.lower()
                    break
        if self.target_index < 0:
            raise ValueError(f"Target {target} doesn't appear in sentence {sentence}")
        else:
            self.context = (
                self.tokens[: self.target_index] + self.tokens[self.target_index + 1 :]
            )

    def change_target(self, new_target, new_pos=None, new_target_index=-1):
        """
        returns a new Sentence instance with the position that is marked as the target changed

        :param new_target: token that should now be the target
        :param new_pos: part of speech of the new target
        :param new_target_index: index of the new target
        """

        new_sent = Sentence(
            sentence=self.sentence,
            target=new_target,
            value=self.value,
            pos=new_pos,
        )
        if new_target_index >= 0:
            new_sent.target_index = new_target_index
        return new_sent

    def replace_target(self, new_target, split_multi_word=True, target_index=-1):
        """
        returns a new Sentence instance where the target is replaced by another word
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
            if target_index >= 0:
                output.target_index = [
                    target_index + i for i in range(len(new_target.split("_")))
                ]
            else:
                output.target_index = [
                    output.target_index + i for i in range(len(new_target.split("_")))
                ]
            output.target = new_target.split("_")
            output.target_token = output.target
        elif target_index >= 0:
            output.target_index = target_index
        return output


class DataSet:
    def __init__(
        self,
        filepath,
        extraction_function,
        use_unsure,
        test_seed,
        test_split_size,
        random_seed,
    ):
        """
        A set of sentences
        :param filepath: where the data that will be converted to Sentences is stored
        :param extraction_function: function that specifically extracts Sentence instances from the provided datafile must have the use_unsure parameter
        :param use_unsure: parameter for the extraction_function that specifies wether the Sentence values are binary or have a third "unsure" bin
        :param test_seed: random seed for extracting the test set
        :param test_split_size: proportion of the extracted test split
        """
        self.sentences = extraction_function(filepath, use_unsure)
        self.seed = test_seed
        self.train_dev_split, self.test_split = self.get_splits(test_split_size)

    def get_splits(self, test_split_size):
        """
        returns train, dev and test splits of the data as a list of three lists
        :param test_split_size: proportion of test split of whole data
        """
        if test_split_size > 1:
            raise ValueError("split size can't be more than 100%")
        num_train_dev = math.floor(len(self.sentences) * (1 - test_split_size))
        num_test = math.floor(len(self.sentences) * test_split_size)
        random.seed(self.seed)
        partitions = random.sample(self.sentences, k=num_train_dev + num_test)
        return (
            partitions[:num_train_dev],
            partitions[num_train_dev : num_train_dev + num_test],
        )

    @staticmethod
    def get_ith_split(i: int, n: int, data: list[Sentence]):
        """
        returns the two splits(train,test) at position i for nfold cross validation

        :param i: which split of the data is extracted(starting with i=0, max n-1)
        :param data: the data that will be split
        """
        if i > n - 1:
            raise ValueError("i can't be bigger than n-1")
        num_sents = len(data)
        num_per_split = math.floor(num_sents / n)
        lower_index = i * num_per_split
        test_split = (
            data[lower_index : lower_index + num_per_split]
            if i < n - 1
            else data[lower_index:]
        )
        train_split = data[:lower_index] + data[lower_index + len(test_split) :]
        return train_split, test_split


class Vectors:
    @staticmethod
    def cos_sim(vec1, vec2):
        """
        simple cosine similarity function
        """
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denominator == 0:
            # breakpoint()
            return 0
        return float(np.dot(vec1, vec2) / denominator)
