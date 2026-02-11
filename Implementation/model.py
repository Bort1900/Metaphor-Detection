from cProfile import label
import random
from turtle import pos
from typing import Self
from sklearn.metrics import confusion_matrix
from data import Vectors
import numpy as np
import time
import torch
from nltk.corpus import stopwords
from torch.nn import CosineSimilarity
from tqdm import tqdm
from data import Sentence, DataSet
from embeddings import Embeddings, BertEmbeddings
import math
from scipy.special import comb
import matplotlib.pyplot as plt


class NThresholdModel:
    def __init__(
        self,
        data: DataSet,
        candidate_source,
        mean_multi_word: bool,
        fit_embeddings: Embeddings,
        score_embeddings: Embeddings,
        use_output_vec: bool,
        apply_candidate_weight: bool,
        restrict_pos=None,
        num_classes=2,
    ):
        """
        Model that categorizes Sentence data in n classes based on n-1 thresholds and uses a candidate set for getting the value
        :param data: Dataset used for training and evaluation
        :param candidate_source: an object with a get_candidate_set function
        :param mean_multi_word: whether embeddings for multi-word tokens should be mean pooled from the embeddings of the individual words
        :param fit_embeddings: source for embeddings for finding best fit candidate
        :param score_embeddings: source for embeddings for scoring for prediction
        :param use_output_vec: whether output vectors(word2vec) should be used for comparing context to candidates
        :param restrict_pos: list of parts of speech to restrict the candidate set to
        :param apply_candidate_weight: whether the candidates should be weighed by association strength, needs SWOWInterface as candidate source
        :param num_classes: number of classes to classify
        """
        self.data = data
        self.train_dev_data = self.data.train_dev_split
        self.test_data = self.data.test_split
        self.candidate_source = candidate_source
        self.mean_multi_word = mean_multi_word
        self.use_output = use_output_vec
        self.fit_embeddings = fit_embeddings
        self.score_embeddings = score_embeddings
        self.decision_thresholds = [0.5 for i in range(num_classes - 1)]
        self.restrict_pos = restrict_pos
        self.num_classes = num_classes
        self.apply_candidate_weight = apply_candidate_weight
        self.stops = stopwords.words("english")
        self.cos = CosineSimilarity(dim=0, eps=1e-6)

    @staticmethod
    def calculate_scores(confusion_matrix: np.ndarray) -> dict[str, float]:
        """
        calculates and returns precision, recall, and f-score metrics for a given evaluation as a dictionary
        :param confusion_matrix: result of the evaluation, prediction vs actual classes
        """
        num_classes = len(confusion_matrix)
        scores = dict()
        for i in range(num_classes):
            if confusion_matrix.sum(1)[i] == 0:
                scores[f"precision_class_{i}"] = 1
            else:
                scores[f"precision_class_{i}"] = float(
                    confusion_matrix[i, i] / confusion_matrix.sum(1)[i]
                )
            if confusion_matrix.sum(0)[i] == 0:
                scores[f"recall_class_{i}"] = 1
            else:
                scores[f"recall_class_{i}"] = float(
                    confusion_matrix[i, i] / confusion_matrix.sum(0)[i]
                )
            if scores[f"precision_class_{i}"] + scores[f"recall_class_{i}"] == 0:
                scores[f"f_1_class_{i}"] = 0
            else:
                scores[f"f_1_class_{i}"] = (
                    2
                    * scores[f"precision_class_{i}"]
                    * scores[f"recall_class_{i}"]
                    / (scores[f"precision_class_{i}"] + scores[f"recall_class_{i}"])
                )

        scores["macro_f_1"] = (
            sum([scores[f"f_1_class_{i}"] for i in range(num_classes)]) / num_classes
        )
        scores["micro_f_1"] = float(
            sum(
                [
                    scores[f"f_1_class_{i}"] * confusion_matrix.sum(0)[i]
                    for i in range(num_classes)
                ]
            )
            / confusion_matrix.sum()
        )
        return scores

    def evaluate(
        self,
        data: list[Sentence] | None = None,
        save_file: str | None = None,
        by_pos: list[str] | None = None,
        by_phrase: bool = False,
    ) -> dict:
        """
        returns a dictionary of recall, precision and f-score metrics after evaluating the model on test data
        :param data: list of sentences for evaluation, defaults to test data
        :param save_file: filepath for possible storing
        :param by_pos: if specified, list of parts of speech, sentences whose target has this pos will be considered for evaluation
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        """
        if not data:
            data = self.test_data
        confusion_matrix = np.zeros([self.num_classes, self.num_classes])
        ignore_count = 0
        for sentence in tqdm(data):
            if by_pos and sentence.pos not in by_pos:
                continue
            try:
                prediction = int(self.predict(sentence, by_phrase=by_phrase))
            except ValueError:
                ignore_count += 1
                continue
            confusion_matrix[prediction, sentence.value] += 1
            # breakpoint()
        print(confusion_matrix)
        print(f"ignored {ignore_count} sentences of {len(data)}")
        scores = NThresholdModel.calculate_scores(confusion_matrix=confusion_matrix)
        if save_file:
            with open(save_file, "w", encoding="utf-8") as output:
                output.write(
                    "Decision thresholds: " + str(self.decision_thresholds)[1:-1] + "\n"
                )
                output.write(str(scores))
        return scores

    def nfold_cross_validate(
        self,
        n: int,
        data: list[Sentence] | None = None,
        save_file: str | None = None,
        by_pos: list[str] | None = None,
        by_phrase: bool = False,
        metrics: list[str] = ["macro_f_1"],
    ):
        """
        performs nfold cross validation for the model and returns the mean evaluation measures

        :param n: number of splits
        :param save_file: filepath for possible storing
        :param by_pos: if specified, list of parts of speech, sentences whose target has this pos will be considered for evaluation
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        :param data: list of sentences to evaluate if specified else model test set
        :param metrics: metric according to which to train the optimum threshold, sum of them if more than 1
        """
        if not data:
            data = self.test_data
        all_scores = dict()
        for i in range(n):
            print(f"Fold {i+1}:")
            train_split, test_split = DataSet.get_ith_split(i, n, data)
            self.train_thresholds(
                data=train_split, by_phrase=by_phrase, metrics=metrics, by_pos=by_pos
            )
            scores = self.evaluate(test_split, by_pos=by_pos, by_phrase=by_phrase)
            if "decision_thresholds" in all_scores:
                all_scores["decision_thresholds"].append(self.decision_thresholds)
            else:
                all_scores["decision_thresholds"] = [self.decision_thresholds]
            for score in scores:
                if score in all_scores:
                    all_scores[score].append(scores[score])
                else:
                    all_scores[score] = [scores[score]]
        output = dict()
        for score in all_scores:
            result = all_scores[score]
            if score == "decision_thresholds":
                if type(result[0][0]) == torch.Tensor:
                    for thresholds in result:
                        for i in range(len(thresholds)):
                            thresholds[i] = thresholds[i].cpu()
                output[score] = {
                    "all": result,
                    "mean": np.mean(result, axis=0),
                    "median": np.median(result, axis=0),
                    "standard_deviation": np.std(result, axis=0),
                }
            else:
                if type(result[0]) == torch.Tensor:
                    result = [value.cpu() for value in result]
                output[score] = {
                    "all": result,
                    "mean": float(np.mean(result)),
                    "median": float(np.median(result)),
                    "standard_deviation": float(np.std(result)),
                }
        if save_file:
            with open(save_file, "w", encoding="utf-8") as write_file:
                write_file.write(str(output))
        return output

    def predict(self, sentence, by_phrase=False):
        """
        returns the class that the model predicts for a given instance
        :param sentence: instance to classify
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        """
        try:
            similarity = self.get_compare_value(sentence, by_phrase=by_phrase)
            # print(similarity)
        except ValueError:
            raise ValueError(f"{sentence.target} not in dictionary")
        scale = self.decision_thresholds + [similarity]
        scale.sort()
        return scale.index(similarity)

    def get_compare_value(self, sentence, by_phrase=False):
        """
        returns the value that is compared with the model's thresholds
        :param sentence: the instance that should be classified
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        """
        predicted_sense = self.best_fit(sentence, by_phrase=by_phrase)
        # print(predicted_sense)
        if predicted_sense == sentence.target:
            return 1
        try:
            target_vector = self.score_embeddings.get_input_vector(
                sentence.target, pos=sentence.pos, exclude_sent=sentence
            )
            if len(predicted_sense.split("_")) > 1 and self.mean_multi_word:
                predicted_vector = self.score_embeddings.get_mean_vector(
                    tokens=predicted_sense.split("_")
                )
            else:
                predicted_vector = self.score_embeddings.get_input_vector(
                    predicted_sense, pos=sentence.pos
                )
        except KeyError:
            raise ValueError(f"{sentence.target} not in dictionary")
        if type(target_vector) == np.ndarray:
            similarity = Vectors.cos_sim(target_vector, predicted_vector)
        elif type(target_vector) == torch.Tensor:
            similarity = self.cos(target_vector, predicted_vector)
        else:
            raise TypeError(
                f"target vector of type {type(target_vector)} not supported"
            )
        if not similarity > 0 and not similarity < 0:
            return 0
        else:
            return similarity

    def best_fit(self, sentence, by_phrase=False):
        """
        returns the best fiting instance from the candidate set to calculate the threshold
        :param sentence: the instance that should be classified
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        """
        candidate_set = self.candidate_source.get_candidate_set(
            sentence.target, pos=self.restrict_pos
        )
        candidate_set.add(sentence.target_token)
        best_similarity = -1
        if by_phrase and sentence.phrase != "unknown":
            context = [word for word in sentence.phrase if word != sentence.target]
        else:
            context = [
                word for word in sentence.context if word.lower() not in self.stops
            ]
        context_vector = self.fit_embeddings.get_mean_vector(context)
        best_candidate = sentence.target
        for candidate in candidate_set:
            if self.use_output:
                try:
                    if len(candidate.split("_")) > 1 and self.mean_multi_word:
                        candidate_vector = self.fit_embeddings.get_mean_vector(
                            tokens=candidate.split("_"), use_output_vecs=True
                        )
                    else:
                        candidate_vector = self.fit_embeddings.get_output_vector(
                            candidate
                        )
                except ValueError:
                    # print(f"Word {candidate} not in dictionary, ignoring candidate")
                    continue
            else:
                try:
                    if len(candidate.split("_")) > 1 and self.mean_multi_word:
                        candidate_vector = self.fit_embeddings.get_mean_vector(
                            tokens=candidate.split("_"), use_output_vecs=False
                        )
                    else:
                        candidate_vector = self.fit_embeddings.get_input_vector(
                            candidate
                        )
                except KeyError:
                    # print(f"Word {candidate} not in dictionary, ignoring candidate")
                    continue
            similarity = Vectors.cos_sim(candidate_vector, context_vector)
            if self.apply_candidate_weight:
                weight = self.candidate_source.get_association_strength(
                    candidate, sentence.target
                )
                similarity *= weight

            if similarity >= best_similarity:
                best_similarity = similarity
                best_candidate = candidate
        return best_candidate

    def train_thresholds(
        self,
        metrics: list[str],
        data: list[Sentence] | None = None,
        by_pos: list[str] | None = None,
        by_phrase: bool = False,
    ):
        """
        trains the model's threshold on the dev_data
        :param data: list of sentences to train on, defaults to dev data
        :param by_pos: if specified, list of parts of speech, sentences whose target has this pos will be considered for evaluation
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        :param metrics: metric according to which to train the optimum threshold, sum of them if more than 1
        """
        if not data:
            data = self.train_dev_data
        scores = dict()
        labels = dict()
        fail_count = 0
        for i, sentence in enumerate(tqdm(data)):
            if by_pos and sentence.pos not in by_pos:
                continue
            try:
                scores[i] = self.get_compare_value(sentence, by_phrase=by_phrase)
                labels[i] = sentence.value
            except ValueError:
                fail_count += 1
                #print(f"{sentence.target} not in dictionary, ignoring sentence")
                continue
        print(f"ignored {fail_count} sentences of {len(data)}")
        sorted_scores = sorted(scores.keys(), key=lambda x: scores[x])
        sorted_labels = [labels[score] for score in sorted_scores]
        all_scores = [scores[score_index] for score_index in sorted_scores]
        sorted_scores_per_class = []
        for i in range(self.num_classes):
            sorted_scores_per_class.append([])
            for score_index in sorted_scores:
                if labels[score_index] == i:
                    sorted_scores_per_class[i].append(scores[score_index])
        last_score = -1
        possible_thresholds = []
        for score in sorted_scores:
            score = scores[score]
            if score != last_score:
                possible_thresholds.append((score + last_score) / 2)
            last_score = score
        best_f_score = 0
        best_threshold = [
            possible_thresholds[i] for i in range(len(self.decision_thresholds))
        ]
        commutation_number: int = int(
            comb(N=len(possible_thresholds), k=len(self.decision_thresholds))
        )
        current_commutation = [i for i in range(len(self.decision_thresholds))]
        for commutation in tqdm(range(commutation_number)):
            current_thresholds = [possible_thresholds[i] for i in current_commutation]
            confusion_matrix = self.get_confusion_matrix(
                scores=sorted_scores_per_class, thresholds=current_thresholds
            )
            threshold_scores = self.calculate_scores(confusion_matrix=confusion_matrix)
            threshold_result = sum([threshold_scores[metric] for metric in metrics])
            if threshold_result > best_f_score:
                best_threshold = current_thresholds
                best_f_score = threshold_result
            if commutation < commutation_number - 1:
                current_commutation = Vectors.get_next_commutation(
                    current_commutation=current_commutation, n=len(possible_thresholds)
                )
        self.decision_thresholds = best_threshold
        print(f"Best Thresholds: {self.decision_thresholds}\nBest score:{best_f_score}")

    @staticmethod
    def get_confusion_matrix(
        scores: list[list[float]], thresholds: list[float]
    ) -> np.ndarray:
        """
        returns the confusion matrix that is given by prediction of classes for the scores with the corresponding labels and given the thresholds

        :param scores: sorted lists of prediction scores per class
        :param thresholds: thresholds to predict classes
        """
        num_classes = len(scores)
        confusion_matrix = np.zeros([num_classes, num_classes])
        for j in range(num_classes):
            indices = [0]
            for i, threshold in enumerate(thresholds):
                length = len(scores[j])
                upper = length - 1
                lower = 0
                while upper - lower >= 1:
                    index = math.ceil((upper + lower) / 2)
                    if scores[j][index - 1] >= threshold:
                        upper = index - 1
                    elif scores[j][index] < threshold:
                        lower = index + 1
                    else:
                        upper = index
                        lower = index
                if lower == length - 1 and scores[j][lower] < threshold:
                    lower = length
                    upper = length
                indices.append(lower)
                confusion_matrix[i, j] = indices[i + 1] - indices[i]
                if i == num_classes - 2:
                    confusion_matrix[i + 1, j] = length - indices[i + 1]
        return confusion_matrix

    def evaluate_per_threshold(
        self,
        start: float,
        steps: int,
        increment: float,
        save_file: str,
        data: list[Sentence] | None = None,
        by_pos: list[str] | None = None,
        by_phrase: bool = False,
    ):
        """
        writes some evaluation metrics into a file after evaluating the model with different thresholds

        :param start: the first threshold to test
        :param steps: the number of thresholds to test
        :param increment: the difference between two thresholds to test
        :param save_file: where to store the results
        :param data: list of sentences to evaluate if specified, else test data of model
        :param by_pos: if specified, list of parts of speech, sentences whose target has this pos will be considered for evaluation
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        """
        if not data:
            data = self.test_data
        if self.num_classes > 2:
            raise ValueError("only works for 2 classes")
        self.decision_thresholds = [start]
        with open(save_file, "w", encoding="utf-8") as output:
            output.write(
                "Threshold\tPrecision\tRecall\tF1(Class 1)\tF1(Class 2)\tF1(Macro-Average)\n"
            )
            for i in range(steps):
                scores = self.evaluate(data, by_pos=by_pos, by_phrase=by_phrase)
                output.write(
                    f'{round(self.decision_thresholds[0],2)}\t{round(scores["precision_class_0"],2)}\t{round(scores["recall_class_0"],2)}\t{round(scores["f_1_class_0"],2)}\t{round(scores["f_1_class_1"],2)}\t{round(scores["macro_f_1"],2)}\n'
                )
                self.decision_thresholds[0] += increment

    def draw_distribution_per_class(
        self,
        save_file: str,
        labels: list[str],
        title: str,
        data: list[Sentence] | None = None,
        by_pos: list[str] | None = None,
        by_phrase: bool = False,
        graph_type: str = "boxplot",
    ):
        """
        draws box plots of the distributions of the prediction scores for each of the classes
        :param labels: labels for each class
        :param title: title of the graph
        :param data: list of sentences to evaluate if specified, else model test set
        :param by_pos: if specified, list of parts of speech, sentences whose target has this pos will be considered for evaluation
        :param save_file: where the plots are stored
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        :param graph_type: boxplot or scatter, how the distribution will be displayed
        """
        colors = [
            "#56B4E9",
            "#E69F00",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
        ]
        if not data:
            data = self.test_data
        datapoints = [[] for _ in range(self.num_classes)]
        for sent in data:
            if by_pos and sent.pos not in by_pos:
                continue
            try:
                similarity = self.get_compare_value(sent, by_phrase=by_phrase)
            except ValueError:
                continue
            if type(similarity) == torch.Tensor:
                datapoints[sent.value].append(similarity.cpu())
            else:
                datapoints[sent.value].append(similarity)
        fig, ax = plt.subplots()
        if graph_type == "boxplot":
            ax.boxplot(datapoints, tick_labels=labels, orientation="horizontal")

            for i, class_data in enumerate(datapoints, start=1):
                med = np.median(class_data)
                ax.text(med, i, f"{med:.2f}", ha="left", va="center")
        elif graph_type == "scatter":
            for i in range(self.num_classes):
                ax.scatter(
                    np.random.rand(len(datapoints[i])) - 0.5,
                    datapoints[i],
                    color=colors[i],
                    label=labels[i],
                    marker="x",
                )

                ax.spines["left"].set_position(("data", 0))
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                plt.xlim(-1, 1)
                plt.xticks([])
                plt.legend()
        else:
            raise ValueError("graph type must be scatter or boxplot")

        plt.title(title)
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()


class MaoModel(NThresholdModel):

    def __init__(
        self,
        data: DataSet,
        candidate_source,
        mean_multi_word: bool,
        fit_embeddings: Embeddings,
        score_embeddings: Embeddings,
        use_output_vec: bool,
        apply_candidate_weight: bool,
        restrict_pos: list[str] | None = None,
    ):
        """
        Model that works like the model from the Mao(2018) paper, see NThresholdModel
        :param data: DataSet instance to train thresholds and evaluate model
        :param candidate_source: an object with a get_candidate_set function
        :param mean_multi_word: whether embeddings for multi-word tokens should be mean pooled from the embeddings of the individual words
        :param fit_embeddings: source for embeddings for finding best fit candidate
        :param score_embeddings: source for embeddings for scoring for prediction
        :param restrict_pos: list of parts of speech the to which candidate sets should be retricted
        :param use_output_vec: whether ouput vectors(word2vec) should be used for comparing context to candidates
        :param apply_candidate_weight: whether the candidates should be weighed by association strength, needs SWOWInterface as candidate source
        """
        super().__init__(
            data=data,
            candidate_source=candidate_source,
            mean_multi_word=mean_multi_word,
            fit_embeddings=fit_embeddings,
            score_embeddings=score_embeddings,
            restrict_pos=restrict_pos,
            use_output_vec=use_output_vec,
            apply_candidate_weight=apply_candidate_weight,
        )


class ContextualMaoModel(NThresholdModel):
    def __init__(
        self,
        data: DataSet,
        candidate_source,
        mean_multi_word: bool,
        fit_embeddings: BertEmbeddings,
        score_embeddings: Embeddings,
        use_context_vec: bool,
        apply_candidate_weight: bool,
        restrict_pos: list[str] | None = None,
        num_classes: int = 2,
    ):
        """
        like Mao Model but uses contextual embeddings
        :param data: DataSet instance to train thresholds and evaluate model
        :param candidate_source: an object with a get_candidate_set function
        :param mean_multi_word: whether embeddings for multi-word tokens should be mean pooled from the embeddings of the individual words
        :param embeddings: source for embeddings for comparing
        :param use_context_vec: whether context vector should be used for comparing context to candidates instead of target word in context
        :param restrict_pos: list of parts of speech to which candidate sets should be retricted
        :param apply_candidate_weight: whether the candidates should be weighed by association strength, needs SWOWInterface as candidate source
        :param num_classes: number of classes to classify
        """
        super().__init__(
            data=data,
            candidate_source=candidate_source,
            mean_multi_word=mean_multi_word,
            fit_embeddings=fit_embeddings,
            score_embeddings=score_embeddings,
            use_output_vec=False,
            restrict_pos=restrict_pos,
            apply_candidate_weight=apply_candidate_weight,
            num_classes=num_classes,
        )
        self.fit_embeddings = fit_embeddings
        self.use_context_vec = use_context_vec

    def get_compare_embedding(self, sentence: Sentence, by_phrase=False):
        """
        returns the vector to which the candidates will be compared

        :param sentence: sentence that will be predicted
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        """
        if by_phrase and sentence.phrase != "unknown":
            phrase = Sentence(sentence=sentence.phrase, target=sentence.target, value=1)
            if self.use_context_vec:
                compare_vector = self.fit_embeddings.get_context_vector(phrase)
            else:
                compare_vector = self.fit_embeddings.get_sentence_vector(phrase)
        else:
            if self.use_context_vec:
                compare_vector = self.fit_embeddings.get_context_vector(sentence)
            else:
                compare_vector = self.fit_embeddings.get_sentence_vector(sentence)
        return compare_vector

    def best_fit(self, sentence: Sentence, by_phrase: bool = False):
        """
        returns the best candidate from the candidate set that fits into the sentence context
        :param sentence: sentence that will be predicted
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        """
        candidate_set = self.candidate_source.get_candidate_set(
            sentence.target, pos=self.restrict_pos
        )
        candidate_set.add(sentence.target_token)
        best_similarity = -1
        compare_vector = self.get_compare_embedding(sentence, by_phrase)
        best_candidate = sentence.target
        for candidate in candidate_set:
            try:
                if len(candidate.split("_")) > 1 and self.mean_multi_word:
                    candidate_vector = self.fit_embeddings.get_mean_vector(
                        tokens=candidate.split("_")
                    )
                else:
                    candidate_vector = self.fit_embeddings.get_input_vector(candidate)
            except ValueError:
                continue
            if self.fit_embeddings.use_phrase_embedding == "concat":
                candidate_vector = torch.concat((candidate_vector, candidate_vector))
            similarity = self.cos(candidate_vector, compare_vector)
            if self.apply_candidate_weight:
                weight = self.candidate_source.get_association_strength(
                    candidate, sentence.target
                )
                similarity *= weight

            if similarity >= best_similarity:
                best_similarity = similarity
                best_candidate = candidate
        return best_candidate


class RandomBaseline(NThresholdModel):
    def __init__(
        self,
        data: DataSet,
        candidate_source,
        score_embeddings: Embeddings,
        restrict_pos: list[str] | None = None,
        num_classes: int = 2,
    ):
        """
        Model that randomly choses a word from the candidate set to predict the class with a threshold
        :param data: DataSet instance to train thresholds and evaluate model
        :param candidate_source: an object with a get_candidate_set function
        :param embeddings: source for embeddings for comparing
        :param restrict_pos: list of parts of speech to which candidate sets should be restricted
        :param num_classes: number of classes for prediction
        """
        super().__init__(
            data=data,
            candidate_source=candidate_source,
            mean_multi_word=False,
            score_embeddings=score_embeddings,
            fit_embeddings=score_embeddings,
            use_output_vec=False,
            restrict_pos=restrict_pos,
            apply_candidate_weight=False,
            num_classes=num_classes,
        )

    def best_fit(self, sentence, by_phrase: bool = False):
        """
        returns random element from the candidate set
        sentence: Sentence instance the model will predict
        :param by_phrase: irrelevant
        """
        candidate_set = self.candidate_source.get_candidate_set(
            sentence.target, pos=self.restrict_pos
        )
        candidate_set.add(sentence.target_token)
        return random.choice(list(candidate_set))


class Models:
    """
    Class for some helper methods
    """

    @staticmethod
    def get_recall_curve(
        data: list[Sentence],
        save_file: str,
        models: list[NThresholdModel],
        graph_labels: list[str],
        by_pos: list[str] | None = None,
        by_phrase: bool = False,
    ):
        """
        draws the false positive rate against true positive rates for multiple binary prediction models against each other

        :param data: data on which to evaluate
        :param save_file: where the plot will be stored
        :param models: list of models to plot against each other
        :param graph_labels: labels for plot of each model
        :param by_pos: if specified, list of parts of speech, sentences whose target has this pos will be considered for evaluation
        :param by_phrase: whether the evaluation will be phrase or sentence based, will default to sentence if phrase is unknown
        """
        fig, ax = plt.subplots()
        for j, model in enumerate(models):
            if model.num_classes != 2:
                raise ValueError("only works for binary models")
            class_counts = [0, 0]
            scores = dict()
            labels = dict()
            for i, sentence in enumerate(tqdm(data)):
                if by_pos and sentence.pos not in by_pos:
                    continue
                try:
                    scores[i] = model.get_compare_value(sentence, by_phrase=by_phrase)
                    class_counts[sentence.value] += 1
                    labels[i] = sentence.value
                except ValueError:
                    #print(f"{sentence.target} not in dictionary, ignoring sentence")
                    continue
            sorted_score_indices = sorted(scores.keys(), key=lambda x: scores[x])
            true_pos_rates = []
            false_pos_rates = []
            true_count = 0
            false_count = 0
            for i, score_index in enumerate(sorted_score_indices):
                label = labels[score_index]
                if label == 1:
                    false_count += 1
                else:
                    true_count += 1
                true_pos_rate = true_count / class_counts[0]
                false_pos_rate = false_count / class_counts[1]
                true_pos_rates.append(true_pos_rate)
                false_pos_rates.append(false_pos_rate)
            plt.plot(false_pos_rates, true_pos_rates, label=graph_labels[j])
        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()
