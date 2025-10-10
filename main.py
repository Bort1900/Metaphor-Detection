from data import DataSet, Sentence
from embeddings import FasttextModel
import pandas as pd
from model import MaoModel
import re
from wordnet_interface import WordNetInterface


def urban_extractor(filepath):
    sentences = []
    fail_counter = 0
    with open(filepath) as data:
        for line in data:
            datapoint = line.split("\t")
            if datapoint[2] != "unsure" and datapoint[1] == datapoint[2]:
                try:
                    if datapoint[2] == "literal":
                        value = 0
                    elif datapoint[2] == "figurative":
                        value = 1
                    else:
                        print(f"{datapoint[2]} is not a valid value")
                        raise ValueError(f"{datapoint[2]} is not a valid value")
                    sentence = Sentence(
                        sentence=datapoint[3],
                        target=datapoint[0].split()[0],
                        value=value,
                        phrase=datapoint[0],
                    )
                    sentences.append(sentence)
                except ValueError:
                    print("Target word not in sentence")
                    fail_counter += 1
    print(len(sentences), fail_counter)
    return sentences


def mohammad_extractor(filepath):
    sentences = []
    fail_counter = 0
    with open(filepath) as data:
        data.readline()
        for line in data:
            datapoint = line.split("\t")
            if len(datapoint) == 5 and float(datapoint[4]) >= 0.7:
                try:
                    if datapoint[3] == "literal":
                        value = 0
                    elif datapoint[3] == "metaphorical":
                        value = 1
                    else:
                        print(f"{datapoint[3]} is not a valid value")
                        raise ValueError(f"{datapoint[3]} is not a valid value")
                    # remove special tokens
                    tokens = re.sub(r"<.*?>", "", datapoint[2])
                    sentence = Sentence(
                        sentence=tokens, target=datapoint[0], value=value
                    )
                    sentences.append(sentence)
                except ValueError:
                    print("Target word not in sentence")
                    fail_counter += 1
    print(len(sentences), fail_counter)
    return sentences


if __name__ == "__main__":
    wn = WordNetInterface()
    urban_dataset = "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English/example_sentences_verb-object.tsv"
    fasttext_dir = "/projekte/semrel/WORK-AREA/Users/navid/wiki.en.bin"
    mohammad_dataset = "/projekte/semrel/WORK-AREA/Users/navid/Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt"
    embeddings = FasttextModel(fasttext_dir)
    data = DataSet(mohammad_dataset, mohammad_extractor)
    train_data, dev_data, test_data = data.get_splits([0, 0.05, 0.95])
    in_out_model = MaoModel(
        dev_data=dev_data,
        test_data=test_data,
        wn=wn,
        embeddings=embeddings,
        use_output_vec=True,
    )
    in_in_model = MaoModel(
        dev_data=dev_data,
        test_data=test_data,
        wn=wn,
        embeddings=embeddings,
        use_output_vec=False,
    )
    in_in_model.find_best_threshold(steps=0.05)
    print(in_in_model.evaluate(in_in_model.test_data))
    in_out_model.find_best_threshold(steps=0.05)
    print(in_out_model.evaluate(in_out_model.test_data))
