from data import DataSet,Sentence,TargetWord
import pandas as pd
from wordnet import WordNetInterface
urban_dataset= "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English/example_sentences_verb-object.tsv"
def urban_extractor(filepath):
    sentences = []
    fail_counter = 0
    with open(filepath) as data:
        for line in data:
            datapoint = line.split("\t")
            if datapoint[2]!="unsure" and datapoint[1]==datapoint[2]:
                try:
                    sentence = Sentence(datapoint[3],datapoint[0].split()[0],datapoint[2],wn)
                    sentences.append(sentence)
                except ValueError:
                    print("Target word not in sentence")
                    fail_counter+=1
    print(len(sentences),fail_counter)
    return sentences

if __name__ == "__main__":
    wn = WordNetInterface()
    DataSet(urban_dataset,urban_extractor)
