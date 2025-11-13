from data import DataSet, Sentence
from embeddings import FasttextModel, WordAssociationEmbeddings, BertEmbeddings
import pandas as pd
from model import MaoModel, ComparingModel, RandomBaseline, ContextualMaoModel
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
            if (datapoint[2] != "unsure" and datapoint[1] == datapoint[2]) or (
                datapoint[2] == "unsure" and use_unsure
            ):
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
                    fail_counter += 1
    print(f"Ignored {fail_counter} of {len(sentences)}")
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
                    fail_counter += 1
    print(f"Ignored {fail_counter} of {len(sentences)}")
    return sentences


if __name__ == "__main__":
    swow = SWOWInterface(number_of_responses=1)
    wn = WordNetInterface(use_pos="v")
    urban_dataset = "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English/example_sentences_verb-object.tsv"
    fasttext_dir = "/projekte/semrel/WORK-AREA/Users/navid/wiki.en.bin"
    fasttext_2_dir = "/projekte/semrel/WORK-AREA/Users/navid/cc.en.300.bin"
    mohammad_dataset = "/projekte/semrel/WORK-AREA/Users/navid/Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt"
    swow_embeddings_r1 = WordAssociationEmbeddings(
        swow=swow,
        index_file="cue_indices_r1.tsv",
        embedding_file="/projekte/semrel/WORK-AREA/Users/navid/graph_embeddings/graph_embeddings_300_r1.npy",
    )
    embeddings = FasttextModel(load_file=fasttext_dir)
    urban_data = DataSet(
        filepath=urban_dataset, extraction_function=urban_extractor, use_unsure=False
    )
    # mohammad_data = DataSet(
    #     filepath=mohammad_dataset,
    #     extraction_function=mohammad_extractor,
    #     use_unsure=False,
    # )
    urban_train_data, urban_dev_data, urban_test_data = urban_data.get_splits(
        splits=[0, 0.1, 0.9]
    )
    # mohammad_train_data, mohammad_dev_data, mohammad_test_data = (
    #     mohammad_data.get_splits(splits=[0, 0.1, 0.9])
    # )
    # random_model = RandomBaseline(
    #     dev_data=urban_dev_data,
    #     test_data=urban_test_data,
    #     candidate_source=wn,
    #     embeddings=embeddings,
    # )
    # random_model.evaluate_per_threshold(
    #     steps=0.1, save_file="model_results/random_baseline_wn.tsv"
    # )
    # random_swow_model = RandomBaseline(
    #     dev_data=urban_dev_data,
    #     test_data=urban_test_data,
    #     candidate_source=swow,
    #     embeddings=swow_embeddings_r1,
    # )
    # random_swow_model.evaluate_per_threshold(
    #     steps=0.1, save_file="model_results/random_baseline_swow_r1.tsv"
    # )
    # comp_model.evaluate_per_threshold(
    #    start=-1,steps=11,increment=0.1, save_file="model_results/comp_model_r1_300.tsv"
    # )
    contextual_embeddings=BertEmbeddings(layer=8)
    contextual_model = ContextualMaoModel(dev_data=urban_dev_data,test_data=urban_test_data,candidate_source=swow,mean_multi_word=True,embeddings=contextual_embeddings,num_classes=2)
    sent=urban_test_data[40]
    print(sent.sentence,sent.target,swow.get_candidate_set(sent.target),contextual_model.best_fit(sent),contextual_model.get_compare_value(sent),contextual_model.predict(sent))
    