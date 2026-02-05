from ast import mod
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
    Models,
)
import re
import numpy as np
import time
import os
from wordnet_interface import WordNetInterface
from swow_interface import SWOWInterface
import torch

# Sentence extractor function for Verb-Object-Dataset (Knuples et al., 2026)
def knuples_extractor(filepath, use_unsure):
    sentences = []
    fail_counter = 0
    not_agree_counter = 0
    with open(filepath) as data:
        i = 0
        for line in data:
            i += 1
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
                not_agree_counter += 1
    print(f"Ignored {fail_counter} bad sentences of {i} ")
    print(
        f"Ignored {not_agree_counter} of {i} because of missing agreement to annotation"
    )
    return sentences

# Sentence extractor function for Wordnet-Verbs-Dataset (Mohammad et al., 2016)
def mohammad_extractor(filepath, use_unsure):
    sentences = []
    fail_counter = 0
    with open(filepath) as data:
        data.readline()
        i = 0
        for line in data:
            i += 1
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
                    else:
                        fail_counter += 1
                        continue
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
    RANDOM_SEED = 53 # seed used in our thesis
    DATA_DIR = "/projekte/semrel/WORK-AREA/Users/navid"
    WORDNET_DIR = "/resources/data/WordNet/WordNet-3.0_extract/"
    knuples_DIR = "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English"
    SWOW_DIR = "/projekte/semrel/WORK-AREA/Users/navid/SWOW-EN18"
    print("loading interfaces")
    # swow_r1 = SWOWInterface(
    #     number_of_responses=1, strength_file="strength.SWOW-EN.R1.20180827.csv"
    # )
    # swow_r1_c2 = SWOWInterface(
    #     number_of_responses=1,
    #     strength_file="strengths_manually_r1.tsv",
    #     candidate_cap=2,
    # )
    # swow_r1_ = SWOWInterface(
    #     number_of_responses=1,
    #     strength_file="strengths_manually_r1.tsv",
    #     candidate_cap=0,
    # )
    swow_r12_ppmi=SWOWInterface(number_of_responses=2,use_ppmi=True,candidate_cap=0,strength_file="strengths_manuall_r12_ppmi.tsv")
    # swow_r123_ppmi=SWOWInterface(number_of_responses=3,strength_file="strengths_manuall_r123_ppmi.tsv",use_ppmi=True)
    # swow_r12 = SWOWInterface(
    #     number_of_responses=2,
    #     strength_file="strengths_manually_r12.tsv",
    #     candidate_cap=0,
    # )
    # swow_r12_c2 = SWOWInterface(
    #     number_of_responses=2,
    #     strength_file="strengths_manually_r12.tsv",
    #     candidate_cap=2,
    # )
    # swow_r123 = SWOWInterface(
    #     number_of_responses=3,
    #     strength_file="strengths_manually_r123.tsv",
    #     candidate_cap=0,
    # )
    # swow_r123_c2 = SWOWInterface(
    #     number_of_responses=3,
    #     strength_file="strengths_manually_r123.tsv",
    #     candidate_cap=2,
    # )
    # swow_r1_ppmi = SWOWInterface(number_of_responses=1, use_ppmi=True)
    wn = WordNetInterface()
    print("loading embeddings")
    fasttext_dir = "/projekte/semrel/WORK-AREA/Users/navid/wiki.en.bin"
    fasttext_2_dir = "/projekte/semrel/WORK-AREA/Users/navid/cc.en.300.bin"
    knuples_dataset = "/projekte/semrel/Annotations/Figurative-Language/multilingual_EN_DE_SI_lit-fig_v-obj_abstract-concrete/English/example_sentences_verb-object.tsv"
    mohammad_dataset = "/projekte/semrel/WORK-AREA/Users/navid/Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt"
    # swow_embeddings_r1 = WordAssociationEmbeddings(
    #     swow=swow_r1,
    #     index_file="cue_indices_manually_r1.tsv",
    #     embedding_file=os.path.join(
    #         DATA_DIR, "graph_embeddings/graph_embeddings_manually_300_r1.npy"
    #     ),
    # )
    swow_embeddings_r12_ppmi=WordAssociationEmbeddings(swow=swow_r12_ppmi,index_file="cue_indices_manually_r12.tsv",embedding_file=os.path.join(DATA_DIR,"graph_embeddings/graph_embeddings_manually_ppmi_300_r12.npy"))
    # WordAssociationEmbeddings.create_graph_embeddings(
    #     swow=swow_r1_ppmi,
    #     index_file="cue_indices_manually_r1.tsv",
    #     embedding_file=os.path.join(
    #         DATA_DIR, "graph_embeddings/graph_embeddings_manually_ppmi_300_r1.npy"
    #     ),
    #     use_only_cues=True,
    #     dimensions=300,
    # )
    # ft_embeddings_wn = FasttextModel(load_file=fasttext_dir, fallback_source=wn)
    # ft_embeddings_swow_r1 = FasttextModel(
    #     load_file=fasttext_dir, fallback_source=swow_r1
    # ) 
    # ft_embeddings_swow_r12 = FasttextModel(
    #     load_file=fasttext_dir, fallback_source=swow_r12
    # )
    # ft_embeddings_swow_r123 = FasttextModel(
    #     load_file=fasttext_dir, fallback_source=swow_r123
    # # )
    # contextual_embeddings_l12_mean = BertEmbeddings(
    #     [9, 10, 11, 12], use_phrase_embedding="mean", mean_weights=[1, 2]
    # )
    # contextual_embeddings_l12_max = BertEmbeddings(
    #     [9, 10, 11, 12], use_phrase_embedding="max", mean_weights=[1, 2]
    # )
    # contextual_embeddings_l12_weighted_12 = BertEmbeddings(
    #     [9, 10, 11, 12], use_phrase_embedding="weighted", mean_weights=[1, 2]
    # )

    # cont_embeddings = BertEmbeddings(layers=[9, 10, 11, 12], use_phrase_embedding=None)
    # n2v_r1_creator=Node2VecEmbeddingsCreator(graph=swow_r1_ppmi,is_directed=False,p=0.5,q=0.5)
    # walks_r1=n2v_r1_creator.simulate_walks(num_walks=15,walk_length=75)
    # n2v_r1_creator.create_embeddings(os.path.join(DATA_DIR,"graph_embeddings/n2v_r1_ppmi_cbow"),walks_r1,dimensions=256,window_size=12,min_count=0,sg=False)
    # n2v_r1_creator.create_embeddings(os.path.join(DATA_DIR,"graph_embeddings/n2v_r1_ppmi_sg"),walks_r1,dimensions=256,window_size=12,min_count=0,sg=True)
    # n2v_r12_creator=Node2VecEmbeddingsCreator(graph=swow_r12_ppmi,is_directed=False,p=0.5,q=0.5)
    # walks_r12=n2v_r12_creator.simulate_walks(num_walks=15,walk_length=75)
    # n2v_r12_creator.create_embeddings(os.path.join(DATA_DIR,"graph_embeddings/n2v_r12_ppmi_cbow"),walks_r12,dimensions=256,window_size=12,min_count=0,sg=False)
    # n2v_r12_creator.create_embeddings(os.path.join(DATA_DIR,"graph_embeddings/n2v_r12_ppmi_sg"),walks_r12,dimensions=256,window_size=12,min_count=0,sg=True)
    # n2v_r123_creator=Node2VecEmbeddingsCreator(graph=swow_r123_ppmi,is_directed=False,p=0.5,q=0.5)
    # walks_r123=n2v_r123_creator.simulate_walks(num_walks=15,walk_length=75)
    # n2v_r123_creator.create_embeddings(os.path.join(DATA_DIR,"graph_embeddings/n2v_r123_ppmi_cbow"),walks_r123,dimensions=256,window_size=12,min_count=0,sg=False)
    # n2v_r123_creator.create_embeddings(os.path.join(DATA_DIR,"graph_embeddings/n2v_r123_ppmi_sg"),walks_r123,dimensions=256,window_size=12,min_count=0,sg=True)

    # n2v_sg = Node2VecEmbeddings(
    #     os.path.join(DATA_DIR, "graph_embeddings/n2v_r1_sg.kv"), swow=swow_r1
    # )
    print("loading datasets")
    knuples_data_unsure = DataSet(
        filepath=knuples_dataset,
        extraction_function=knuples_extractor,
        use_unsure=False,
        test_seed=RANDOM_SEED,
        test_split_size=0.2,
    )
    knuples_data_unsure = DataSet(
        filepath=knuples_dataset,
        extraction_function=knuples_extractor,
        use_unsure=True,
        test_seed=RANDOM_SEED,
        test_split_size=0.2,
    )
    # mohammad_data = DataSet(
    #     filepath=mohammad_dataset,
    #     extraction_function=mohammad_extractor,
    #     use_unsure=False,
    #     test_seed=RANDOM_SEED,
    #     test_split_size=0.2,
    # )
    print("loading models")
    swow_embeddings_matrix=NThresholdModel(data=knuples_data_unsure,candidate_source=wn,mean_multi_word=False,fit_embeddings=swow_embeddings_r12_ppmi,score_embeddings=swow_embeddings_r12_ppmi,use_output_vec=False,apply_candidate_weight=False,restrict_pos=["v"],num_classes=3)

    # swow_candidates_r1_mean_phrase_bare = ContextualMaoModel(
    #     data=knuples_data,
    #     candidate_source=swow_r1,
    #     mean_multi_word=True,
    #     fit_embeddings=contextual_embeddings_l12_mean,
    #     score_embeddings=cont_embeddings,
    #     use_context_vec=False,
    #     apply_candidate_weight=False,
    #     restrict_pos=None,
    # )
    # swow_candidates_r1_mean_phrase_c2 = ContextualMaoModel(
    #     data=knuples_data,
    #     candidate_source=swow_r1_c2,
    #     mean_multi_word=True,
    #     fit_embeddings=contextual_embeddings_l12_mean,
    #     score_embeddings=cont_embeddings,
    #     use_context_vec=False,
    #     apply_candidate_weight=False,
    #     restrict_pos=None,
    # )
    # swow_candidates_r1_mean_phrase_weighted = ContextualMaoModel(
    #     data=knuples_data,
    #     candidate_source=swow_r1_c2,
    #     mean_multi_word=True,
    #     fit_embeddings=contextual_embeddings_l12_mean,
    #     score_embeddings=cont_embeddings,
    #     use_context_vec=False,
    #     apply_candidate_weight=True,
    #     restrict_pos=None,
    # )
    # swow_candidates_r1_mean_phrase_restricted = ContextualMaoModel(
    #     data=knuples_data,
    #     candidate_source=swow_r1_c2,
    #     mean_multi_word=True,
    #     fit_embeddings=contextual_embeddings_l12_mean,
    #     score_embeddings=cont_embeddings,
    #     use_context_vec=False,
    #     apply_candidate_weight=False,
    #     restrict_pos=["v"],
    # )
    # swow_candidates_r1_mean_phrase_weighted_restricted = ContextualMaoModel(
    #     data=knuples_data,
    #     candidate_source=swow_r1_c2,
    #     mean_multi_word=True,
    #     fit_embeddings=contextual_embeddings_l12_mean,
    #     score_embeddings=cont_embeddings,
    #     use_context_vec=False,
    #     apply_candidate_weight=True,
    #     restrict_pos=["v"],
    # )
    # random_baseline = RandomBaseline(
    #     data=knuples_data,
    #     candidate_source=swow_r1_c2,
    #     score_embeddings=ft_embeddings_swow_r1,
    #     restrict_pos=["v"],
    # )

    # models = [
    #     swow_candidates_r1_mean_phrase_bare,
    #     swow_candidates_r1_mean_phrase_c2,
    #     swow_candidates_r1_mean_phrase_weighted,
    #     swow_candidates_r1_mean_phrase_restricted,
    #     swow_candidates_r1_mean_phrase_weighted_restricted,
    #     random_baseline,
    # ]
    # filenames = [
    #     "SWOW Candidates R1 mean phrase bare",
    #     "SWOW Candidates R1 mean phrase cap",
    #     "SWOW Candidates R1 mean phrase weighted",
    #     "SWOW Candidates R1 mean phrase restricted",
    #     "SWOW Candidates R1 mean phrase weighted and restricted",
    #     "Random Baseline",
    # ]
    models=[swow_embeddings_matrix]
    filenames=["test"]
    print("evaluating")
    for model, name in zip(models, filenames):
        print(name)
        model.nfold_cross_validate(
            n=5,
            data=model.train_dev_data,
            save_file="model_evaluations/" + name + ".txt",
            by_pos=["v"],
        )
        model.draw_distribution_per_class(
            save_file="model_distributions/" + name + ".png",
            title=name,
            data=model.train_dev_data,
            by_pos=["v"],
            labels=["metaphorical", "literal"],
        )
    Models.get_recall_curve(
        data=models[0].train_dev_data,
        save_file="recall_curves/" + "_".join(filenames) + ".png",
        models=models,
        graph_labels=filenames,
        by_pos=["v"],
    )
    # wn_candidates_best.nfold_cross_validate(
    #     data=wn_candidates_best.train_dev_data,
    #     n=10,
    #     by_pos=["v"],
    #     exclude_extremes=(-1, 1),
    #     save_file="test1.txt",
    #     training_epochs=4,
    # )
    # wn_candidates_best.nfold_cross_validate(
    #     data=wn_candidates_best.train_dev_data,
    #     n=10,
    #     by_pos=["v"],
    #     exclude_extremes=(-1, 1),
    #     save_file="test2.txt",
    #     training_epochs=5,
    # )
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
    # wn_candidates_best.decision_thresholds = [0.475]
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
