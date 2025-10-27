from data import DataSet, Sentence
from embeddings import FasttextModel, WordAssociationEmbeddings
import pandas as pd
from model import MaoModel
import re
import numpy as np
import time
from wordnet_interface import WordNetInterface
from swow_interface import SWOWInterface


def urban_extractor(filepath, use_unsure):
    sentences = []
    fail_counter = 0
    with open(filepath) as data:
        for line in data:
            datapoint = line.split("\t")
            if (datapoint[2] != "unsure" and datapoint[1] == datapoint[2]) or(datapoint[2]=="unsure" and use_unsure):
                try:
                    if datapoint[2] == "unsure":
                        value = 1
                    elif datapoint[2] == "literal" and use_unsure:
                        value = 2
                    elif datapoint[2] == "literal":
                        value = 1
                    elif datapoint[2] == "figurative":
                        value = 0
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


def mohammad_extractor(filepath, use_unsure):
    sentences = []
    fail_counter = 0
    with open(filepath) as data:
        data.readline()
        for line in data:
            datapoint = line.split("\t")
            if len(datapoint) == 5 and (float(datapoint[4]) >= 0.7 or use_unsure):
                if float(datapoint[4]) < 0.7:
                    value = 1
                else:
                    if datapoint[3] == "literal" and use_unsure:
                        value = 2
                    elif datapoint[3] == "literal":
                        value = 1
                    elif datapoint[3] == "metaphorical":
                        value = 0
                try:
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
    moment = time.time()
    print("loading small world of words")
    swow = SWOWInterface(number_of_responses=1)
    print(time.time() - moment)
    print("loading wordnet")
    moment = time.time()
    wn = WordNetInterface(use_pos="v")
    # print(time.time() - moment)
    # print("loading SWOW graph embeddings")
    # moment = time.time()
    urban_dataset = "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English/example_sentences_verb-object.tsv"
    fasttext_dir = "/projekte/semrel/WORK-AREA/Users/navid/wiki.en.bin"
    fasttext_2_dir = "/projekte/semrel/WORK-AREA/Users/navid/cc.en.300.bin"
    mohammad_dataset = "/projekte/semrel/WORK-AREA/Users/navid/Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt"
    # swow_embeddings = WordAssociationEmbeddings(
    #     swow=swow,
    #     index_file="cue_indices.tsv",
    #     embedding_file="graph_embeddings_300.npy",
    # )
    print(time.time() - moment)
    print("loading Fasttext Embeddings")
    moment = time.time()
    embeddings = FasttextModel(load_file=fasttext_dir)
    print(time.time() - moment)
    print("preparing data")
    moment = time.time()
    data = DataSet(filepath=mohammad_dataset, extraction_function=mohammad_extractor)
    train_data, dev_data, test_data = data.get_splits(splits=[0, 0, 1])
    print(time.time() - moment)
    print("creating models")
    moment = time.time()
    wn_mao_model = MaoModel(
        dev_data=dev_data,
        test_data=test_data,
        candidate_source=swow,
        embeddings=embeddings,
        use_output_vec=False,
    )
    # swow_mao_model = MaoModel(
    #     dev_data=dev_data,
    #     test_data=test_data,
    #     candidate_source=swow,
    #     candidates_by_pos=False,
    #     embeddings=embeddings,
    #     use_output_vec=False,
    # )
    # print(time.time() - moment)
    # print("training wordnet model")
    # moment = time.time()
    # wn_mao_model.find_best_threshold(steps=0.05)

    # print(time.time() - moment)
    # print("training SWOW model")
    # moment = time.time()
    # swow_mao_model.find_best_threshold(steps=0.05)

    print(time.time() - moment)
    print("evaluating wordnet model")
    moment = time.time()
    print(
        wn_mao_model.evaluate_per_threshold(
            steps=0.1, save_file="model_results/swow_candidate_set_r1_mohammad.tsv"
        )
    )
    print(time.time() - moment)
    # print("evaluating SWOW model")
    # moment = time.time()
    # print(swow_mao_model.evaluate(data=swow_mao_model.test_data))
