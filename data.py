import wordnet
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random
import math

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
    
    def get_splits(self,splits):
        '''
            returns train, dev and test splits of the data, proportions can be given in this order
        '''
        if len(splits)!= 3:
            raise ValueError("Please give one value for train, development and test split each")
        if sum(splits)>1:
            raise ValueError("Splits mustn't add to more than 100%")    
        num_train = math.floor(len(self.sentences)*splits[0])
        num_dev = math.floor(len(self.sentences)*splits[1])
        num_test = math.floor(len(self.sentences)*splits[2])
        partitions = random.sample(self.sentences,k=num_train+num_dev+num_test)
        return partitions[:num_train],partitions[num_train:num_train+num_dev],partitions[num_train+num_dev:num_train+num_dev+num_test]