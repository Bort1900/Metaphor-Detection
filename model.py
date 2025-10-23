import random
from data import Vectors
import numpy as np
import time
from tqdm import tqdm
import math


class MaoModel:
    def __init__(
        self,
        dev_data,
        test_data,
        candidate_source,
        candidates_by_pos,
        embeddings,
        use_output_vec,
    ):
        self.dev_data = dev_data
        self.test_data = test_data
        self.candidate_source = candidate_source
        self.use_pos = candidates_by_pos
        self.use_output = use_output_vec
        self.embeddings = embeddings
        self.decision_threshold = 0.6

    def train_threshold(self, increment, epochs, batch_size=-1):
        """
        looks for optimal threshold by approximating recall to precision
        increment: initial threshold change when approximating
        epochs: number of times to go over development data
        batch_size: number of datapoints after which threshold is aligned, if -1 the dataset will not be separated into batches
        """

        if batch_size < 0:
            batch_size = len(self.dev_data)
        else:
            batch_number = math.floor(len(self.dev_data) / batch_size)
        alternating_counter = 0  # checks for jumping over optimum
        for i in range(epochs):
            random.shuffle(self.dev_data)
            print(f"epoch {i+1}:")
            batch_start = 0
            for _ in range(batch_number):
                scores = self.evaluate(
                    self.dev_data[batch_start : batch_start + batch_size]
                )[0]
                print(
                    f"Current_threshold: {self.decision_threshold}\nBatch F-score: {scores["micro_f_1"]}"
                )
                batch_start += batch_size
                if scores["recall"] < scores["precision"]:
                    self.decision_threshold += increment
                    alternating_counter = max(0, alternating_counter * (-1) + 1)
                elif scores["precision"] < scores["recall"]:
                    self.decision_threshold -= increment
                    alternating_counter = min(0, alternating_counter * (-1) - 1)
                # 4 alternations between raising and lowering threshold
                if abs(alternating_counter) >= 4:
                    increment /= 2
                    alternating_counter = 0
                print(alternating_counter)

    def optimize_threshold(self, max_epochs=100):
        increment = 0.1
        self.decision_threshold = 0
        direction = 1  # 1:upwards, -1 downwards
        lower_bound = 0
        i = 0
        upper_bound = 1
        last_f_score = self.evaluate(self.dev_data)[0]["micro_f_1"]
        while (
            self.decision_threshold <= upper_bound
            and self.decision_threshold >= lower_bound
            and i < max_epochs
        ):
            self.decision_threshold += direction * increment
            current_f_score = self.evaluate(self.dev_data)[0]["micro_f_1"]
            if current_f_score < last_f_score:
                if direction == 1:
                    upper_bound = self.decision_threshold
                else:
                    lower_bound = self.decision_threshold
                increment /= 2
                direction *= -1
            last_f_score = current_f_score
            print(self.decision_threshold, last_f_score, lower_bound, upper_bound)
            i += 1

    def find_best_threshold(self, steps):
        self.decision_threshold = 0
        best_threshold = 0
        best_f_score = 0
        while self.decision_threshold < 1:
            f_score = self.evaluate(self.dev_data)[0]["micro_f_1"]
            if f_score > best_f_score:
                best_f_score = f_score
                best_threshold = self.decision_threshold
            self.decision_threshold += steps
        self.decision_threshold = best_threshold
        print(f"Best Threshold: {self.decision_threshold}, F-Score: {best_f_score}")

    @staticmethod
    def calculate_scores(confusion_matrix):
        scores = dict()
        if confusion_matrix.sum(1)[0] == 0:
            scores["anti_precision"] = 1
        else:
            scores["anti_precision"] = float(
                confusion_matrix[0, 0] / confusion_matrix.sum(1)[0]
            )
        if confusion_matrix.sum(1)[1] == 0:
            scores["precision"] = 1
        else:
            scores["precision"] = float(
                confusion_matrix[1, 1] / confusion_matrix.sum(1)[1]
            )
        if confusion_matrix.sum(0)[1] == 0:
            scores["recall"] = 1
        else:
            scores["recall"] = float(
                confusion_matrix[1, 1] / confusion_matrix.sum(0)[1]
            )
        if confusion_matrix.sum(0)[0] == 0:
            scores["anti_recall"] = 1
        else:
            scores["anti_recall"] = float(
                confusion_matrix[0, 0] / confusion_matrix.sum(0)[0]
            )
        if scores["precision"] + scores["recall"] == 0:
            scores["f_1"] = 0
        else:
            scores["f_1"] = (
                2
                * scores["precision"]
                * scores["recall"]
                / (scores["precision"] + scores["recall"])
            )
        if scores["anti_precision"] + scores["anti_recall"] == 0:
            scores["anti_f_1"] = 0
        else:
            scores["anti_f_1"] = (
                2
                * scores["anti_precision"]
                * scores["anti_recall"]
                / (scores["anti_precision"] + scores["anti_recall"])
            )
        scores["macro_f_1"] = (scores["f_1"] + scores["anti_f_1"]) / 2
        scores["micro_f_1"] = float(
            (
                scores["f_1"] * confusion_matrix.sum(0)[1]
                + scores["anti_f_1"] * confusion_matrix.sum(0)[0]
            )
            / confusion_matrix.sum()
        )
        return scores

    def evaluate(self, data):
        confusion_matrix = np.zeros([2, 2])
        fp_indices = []
        fn_indices = []
        ignore_count = 0
        for i, sentence in enumerate(tqdm(data)):
            try:
                prediction = int(self.predict(sentence))
            except ValueError:
                print(f"{sentence.target} not in dictionary, ignoring sentence")
                ignore_count += 1
                continue
            if prediction > sentence.value:
                fp_indices.append(i)
            elif prediction < sentence.value:
                fn_indices.append(i)
            confusion_matrix[prediction, sentence.value] += 1
        print(confusion_matrix)
        print(f"ignored {ignore_count} sentences of {len(data)}")
        scores = MaoModel.calculate_scores(confusion_matrix=confusion_matrix)
        return scores, fp_indices, fn_indices

    def predict(self, sentence):
        predicted_sense = self.best_fit(sentence)
        try:
            target_vector = self.embeddings.get_input_vector(sentence.target)
        except KeyError:
            raise ValueError(f"{sentence.target} not in dictionary")
        predicted_vector = self.embeddings.get_input_vector(predicted_sense)
        similarity = Vectors.cos_sim(target_vector, predicted_vector)
        return similarity < self.decision_threshold

    def best_fit(self, sentence):
        if self.use_pos:
            candidate_set = self.candidate_source.get_candidate_set(
                sentence.target, self.use_pos
            )
        else:
            candidate_set = self.candidate_source.get_candidate_set(sentence.target)
        candidate_set.add(sentence.target_token)
        best_similarity = -1
        context_vector = self.embeddings.get_mean_vector(sentence.context)
        best_candidate = sentence.target
        for candidate in candidate_set:
            if self.use_output:
                try:
                    candidate_vector = self.embeddings.get_output_vector(candidate)
                except ValueError:
                    print(f"Word {candidate} not in dictionary, ignoring candidate")
                    continue
            else:
                try:
                    candidate_vector = self.embeddings.get_input_vector(candidate)
                except KeyError:
                    # print(f"Word {candidate} not in dictionary, ignoring candidate")
                    continue
            similarity = Vectors.cos_sim(candidate_vector, context_vector)
            if similarity >= best_similarity:
                best_similarity = similarity
                best_candidate = candidate
        return best_candidate
