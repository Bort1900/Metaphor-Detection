from data import DataSet, Sentence
from embeddings import FasttextModel, WordAssociationEmbeddings, BertEmbeddings
import pandas as pd
from model import MaoModel, ComparingModel, RandomBaseline, ContextualMaoModel
import re
import numpy as np
import time
import os
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
                    verb_sentence = Sentence(
                        sentence=datapoint[3],
                        target=datapoint[0].split()[0],
                        value=value,
                        phrase=datapoint[0],
                        pos="v",
                    )
                    noun_sentence = Sentence(
                        sentence=datapoint[3],
                        target=datapoint[0].split()[1],
                        value=value,
                        phrase=datapoint[0],
                        pos="n",
                    )
                    sentences += [verb_sentence, noun_sentence]
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
                        sentence=tokens, target=datapoint[0], value=value, pos="v"
                    )
                    sentences.append(sentence)
                except ValueError:
                    fail_counter += 1
    print(f"Ignored {fail_counter} of {len(sentences)}")
    return sentences


if __name__ == "__main__":
    DATA_DIR = "/projekte/semrel/WORK-AREA/Users/navid"
    WORDNET_DIR = "/resources/data/WordNet/WordNet-3.0_extract/"
    URBAN_DIR = "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English"
    SWOW_DIR = "/projekte/semrel/WORK-AREA/Users/navid/SWOW-EN18"
    print("loading interfaces")
    # swow_r1 = SWOWInterface(
    #     number_of_responses=1, strength_file="strength.SWOW-EN.R1.20180827.csv"
    # )
    # swow_r12 = SWOWInterface(
    #     number_of_responses=2, strength_file="strengths_manually_r12.tsv"
    # )
    # swow_r123 = SWOWInterface(
    #     number_of_responses=3, strength_file="strength.SWOW-EN.R123.20180827.csv"
    # )
    wn = WordNetInterface()
    print("loading embeddings")
    urban_dataset = "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English/example_sentences_verb-object.tsv"
    fasttext_dir = "/projekte/semrel/WORK-AREA/Users/navid/wiki.en.bin"
    fasttext_2_dir = "/projekte/semrel/WORK-AREA/Users/navid/cc.en.300.bin"
    mohammad_dataset = "/projekte/semrel/WORK-AREA/Users/navid/Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt"
    # swow_embeddings_r1 = WordAssociationEmbeddings(
    #     swow=swow_r1,
    #     index_file="cue_indices_r1.tsv",
    #     embedding_file="/projekte/semrel/WORK-AREA/Users/navid/graph_embeddings/graph_embeddings_300_r1.npy",
    # )
    # swow_embeddings_r123 = WordAssociationEmbeddings(
    #     swow=swow_r123,
    #     index_file="cue_indices.tsv",
    #     embedding_file=os.path.join(
    #         DATA_DIR, "graph_embeddings/graph_embeddings_300.npy"
    #     ),
    # )

    # swow_embeddings_r12 = WordAssociationEmbeddings(
    #     swow=swow_r12,
    #     index_file="cue_indices_manually_r12.tsv",
    #     embedding_file="/projekte/semrel/WORK-AREA/Users/navid/graph_embeddings/graph_embeddings_manually_300_r12.npy",
    # )
    # swow_embeddings_r123 = WordAssociationEmbeddings(
    #     swow=swow_r123,
    #     index_file="cue_indices_manually_r123.tsv",
    #     embedding_file="/projekte/semrel/WORK-AREA/Users/navid/graph_embeddings/graph_embeddings_manually_300_r123.npy",
    # )
    # embeddings = FasttextModel(load_file=fasttext_dir)
    contextual_embeddings_l8 = BertEmbeddings([8])
    contextual_embeddings_l12 = BertEmbeddings([12])
    contextual_embeddings_l9_12 = BertEmbeddings([9, 10, 11, 12])
    print("loading datasets")
    urban_data = DataSet(
        filepath=urban_dataset, extraction_function=urban_extractor, use_unsure=False
    )
    mohammad_data = DataSet(
        filepath=mohammad_dataset,
        extraction_function=mohammad_extractor,
        use_unsure=False,
    )
    print("loading models")
    wn_bert = ContextualMaoModel(
        data=urban_data,
        candidate_source=wn,
        mean_multi_word=True,
        fit_embeddings=contextual_embeddings_l8,
        score_embeddings=contextual_embeddings_l8,
        restrict_pos=True,
    )
    moment = time.time()
    for i in range(10, 20):
        wn_bert.predict(wn_bert.test_data[i])
        print("took", time.time() - moment)
        moment = time.time()
    # print("loading models")
    # swow_model_r1=MaoModel(dev_data=urban_dev_data,test_data=urban_test_data,candidate_source=swow_r1,mean_multi_word=False,embeddings=swow_embeddings_r1,use_output_vec=False)
    # swow_model_r12=MaoModel(dev_data=urban_dev_data,test_data=urban_test_data,candidate_source=swow_r12,mean_multi_word=False,embeddings=swow_embeddings_r12,use_output_vec=False)
    # swow_model_r123=MaoModel(dev_data=urban_dev_data,test_data=urban_test_data,candidate_source=swow_r123,mean_multi_word=False,embeddings=swow_embeddings_r123,use_output_vec=False)
    # print("evaluating")
    # swow_model_r1.evaluate_per_threshold(start=0,steps=11,increment=0.1,save_file="model_results/swow_model_r1_manual.tsv")
    # swow_model_r12.evaluate_per_threshold(start=0,steps=11,increment=0.1,save_file="model_results/swow_model_r12_manual.tsv")
    # swow_model_r123.evaluate_per_threshold(start=0,steps=11,increment=0.1,save_file="model_results/swow_model_r123_manual.tsv")
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
    # )
    # comp_model=ComparingModel(dev_data=mohammad_dev_data,test_data=mohammad_test_data,literal_embeddings=embeddings,associative_embeddings=swow_embeddings_r1,num_classes=2,use_output_vec=False)
    # comp_model.evaluate_per_threshold(
    #    start=0,steps=11,increment=0.1, save_file="model_results/comp_model_r1_300.tsv"
    # )
    # contextual_embeddings=BertEmbeddings(layer=8)
    # contextual_model = ContextualMaoModel(dev_data=urban_dev_data,test_data=urban_test_data,candidate_source=swow,mean_multi_word=True,embeddings=contextual_embeddings,num_classes=2)
    # sent=urban_test_data[40]
    # print(sent.sentence,sent.target,swow.get_candidate_set(sent.target),contextual_model.best_fit(sent),contextual_model.get_compare_value(sent),contextual_model.predict(sent))
