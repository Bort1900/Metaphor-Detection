from data import DataSet,Sentence
from embeddings import FasttextModel
import pandas as pd
from model import MaoModel
import time
from wordnet_interface import WordNetInterface
def urban_extractor(filepath):
    sentences = []
    fail_counter = 0
    with open(filepath) as data:
        for line in data:
            datapoint = line.split("\t")
            if datapoint[2]!="unsure" and datapoint[1]==datapoint[2]:
                try:
                    sentence = Sentence(datapoint[3],datapoint[0].split()[0],datapoint[2])
                    sentences.append(sentence)
                except ValueError:
                    print("Target word not in sentence")
                    fail_counter+=1
    print(len(sentences),fail_counter)
    return sentences

if __name__ == "__main__":
    wn = WordNetInterface()
    urban_dataset= "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English/example_sentences_verb-object.tsv"
    fasttext_dir="/projekte/semrel/WORK-AREA/Users/navid/wiki.en.bin"
    embeddings = FasttextModel(fasttext_dir)
    data = DataSet(urban_dataset,urban_extractor)
    train_data,dev_data,test_data = data.get_splits([0,0.05,0.95])
    model = MaoModel(dev_data,test_data,wn,embeddings)
    print(dev_data[0].tokens,dev_data[0].target)
    print(model.best_fit(dev_data[0],True))
    