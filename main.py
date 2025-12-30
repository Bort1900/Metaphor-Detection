from data import DataSet, Sentence
from embeddings import (
    FasttextModel,
    WordAssociationEmbeddings,
    BertEmbeddings,
    Node2VecEmbeddings,
    Node2VecEmbeddingsCreator,
)
import pandas as pd
from model import (
    MaoModel,
    ComparingModel,
    RandomBaseline,
    ContextualMaoModel,
    NThresholdModel,
)
import re
import numpy as np
import time
import os
from wordnet_interface import WordNetInterface
from swow_interface import SWOWInterface
import torch


def urban_extractor(filepath, use_unsure):
    sentences = []
    fail_counter = 0
    not_agree_counter=0
    with open(filepath) as data:
        i=0
        for line in data:
            i+=1
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
                except ValueError as e:
                    fail_counter += 1
            else:
                not_agree_counter+=1
    print(f"Ignored {fail_counter} bad sentences of {i} ")
    print(f'Ignored {not_agree_counter} of {i} because of missing agreement to annotation')
    return sentences


def mohammad_extractor(filepath, use_unsure):
    sentences = []
    fail_counter = 0
    with open(filepath) as data:
        data.readline()
        i=0
        for line in data:
            i+=1
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
    print(f"Ignored {fail_counter} of {i}")
    return sentences


if __name__ == "__main__":
    RANDOM_SEED = 153
    DATA_DIR = "/projekte/semrel/WORK-AREA/Users/navid"
    WORDNET_DIR = "/resources/data/WordNet/WordNet-3.0_extract/"
    URBAN_DIR = "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English"
    SWOW_DIR = "/projekte/semrel/WORK-AREA/Users/navid/SWOW-EN18"
    print(torch.cuda.get_device_name())
    print("loading interfaces")
    swow_r1 = SWOWInterface(
        number_of_responses=1, strength_file="strength.SWOW-EN.R1.20180827.csv"
    )
    swow_r1_c2 = SWOWInterface(
        number_of_responses=1,
        strength_file="strength.SWOW-EN.R1.20180827.csv",
        candidate_cap=2,
    )
    swow_r1_man = SWOWInterface(
        number_of_responses=1,
        strength_file="strengths_manually_r1.tsv",
        candidate_cap=0,
    )
    wn = WordNetInterface()
    print("loading embeddings")
    urban_dataset = "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English/example_sentences_verb-object.tsv"
    fasttext_dir = "/projekte/semrel/WORK-AREA/Users/navid/wiki.en.bin"
    fasttext_2_dir = "/projekte/semrel/WORK-AREA/Users/navid/cc.en.300.bin"
    mohammad_dataset = "/projekte/semrel/WORK-AREA/Users/navid/Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt"
    swow_embeddings_r1 = WordAssociationEmbeddings(
        swow=swow_r1,
        index_file="cue_indices_r1.tsv",
        embedding_file="/projekte/semrel/WORK-AREA/Users/navid/graph_embeddings/graph_embeddings_300_r1.npy",
    )
    ft_embeddings = FasttextModel(load_file=fasttext_dir)
    cont_embeddings_max = BertEmbeddings(
        layers=[9, 10, 11, 12], use_phrase_embedding="max"
    )
    cont_embeddings_mean = BertEmbeddings(
        layers=[9, 10, 11, 12], use_phrase_embedding="mean"
    )
    cont_embeddings = BertEmbeddings(layers=[9, 10, 11, 12], use_phrase_embedding=None)
    n2v_sg = Node2VecEmbeddings(
        os.path.join(DATA_DIR, "graph_embeddings/n2v_r1_sg.kv"), swow=swow_r1_man
    )
    print("loading datasets")
    urban_data = DataSet(
        filepath=urban_dataset,
        extraction_function=urban_extractor,
        use_unsure=True,
        seed=RANDOM_SEED,
    )
    mohammad_data = DataSet(
        filepath=mohammad_dataset,
        extraction_function=mohammad_extractor,
        use_unsure=True,
        seed=RANDOM_SEED,
    )
    print("loading models")
    wn_baseline_best = ContextualMaoModel(
        data=urban_data,
        candidate_source=wn,
        mean_multi_word=False,
        fit_embeddings=cont_embeddings_max,
        score_embeddings=cont_embeddings_max,
        use_context_vec=True,
        apply_candidate_weight=False,
        restrict_pos=["v"],
        num_classes=3,
    )
    swow_candidates_best = ContextualMaoModel(
        data=urban_data,
        candidate_source=swow_r1_c2,
        mean_multi_word=False,
        fit_embeddings=cont_embeddings_mean,
        score_embeddings=cont_embeddings_mean,
        use_context_vec=False,
        apply_candidate_weight=True,
        restrict_pos=["v"],
        num_classes=3,
    )
    swow_model_best = NThresholdModel(
        data=urban_data,
        candidate_source=swow_r1,
        mean_multi_word=False,
        fit_embeddings=swow_embeddings_r1,
        score_embeddings=swow_embeddings_r1,
        use_output_vec=False,
        apply_candidate_weight=False,
        restrict_pos=None,
        num_classes=3,
    )
    wn_candidates_best = NThresholdModel(
        data=urban_data,
        candidate_source=wn,
        mean_multi_word=False,
        fit_embeddings=n2v_sg,
        score_embeddings=n2v_sg,
        use_output_vec=False,
        apply_candidate_weight=False,
        restrict_pos=["v"],
        num_classes=3,
    )
    print("evaluating")
    # models = [
    #     wn_baseline_best,
    #     swow_candidates_best,
    #     swow_model_best,
    #     wn_candidates_best,
    # ]
    # names = [
    #     "wn_baseline_best",
    #     "swow_candidates_best",
    #     "swow_model_best",
    #     "wn_candidates_best",
    # ]
    # for model, name in zip(models, names):
    #     model.train_thresholds(
    #         increment=0.01, epochs=5, by_pos=["v"], exclude_extremes=[-1, 1]
    #     )
    #     print(model.evaluate(by_pos=["v"]))
        # model.nfold_cross_validate(
        #     n=10,
        #     save_file="model_results/" + name + ".txt",
        #     by_pos=["v"],
        #     exclude_extremes=[-1, 1],     
        # )
        # model.draw_distribution_per_class(
        #     save_file="model_distributions/" + name + ".png",
        #     by_pos=["v"],
        #     labels=["metaphorical", "literal"],
        #     title=name,
        # )
        # model.evaluate_per_threshold(start=0.1,increment=0.05,steps=11,save_file="model_evaluations/"+name+".txt",by_pos=["v"])
