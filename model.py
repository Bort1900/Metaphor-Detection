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
                precision, recall, f_score = self.evaluate(
                    self.dev_data[batch_start : batch_start + batch_size]
                )
                print(
                    f"Current_threshold: {self.decision_threshold}\nBatch F-score: {f_score}"
                )
                batch_start += batch_size
                if recall < precision:
                    self.decision_threshold += increment
                    alternating_counter = max(0, alternating_counter * (-1) + 1)
                elif precision < recall:
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
        last_f_score = self.evaluate(self.dev_data)[2]
        while (
            self.decision_threshold <= upper_bound
            and self.decision_threshold >= lower_bound
            and i < max_epochs
        ):
            self.decision_threshold += direction * increment
            current_f_score = self.evaluate(self.dev_data)[2]
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
            f_score = self.evaluate(self.dev_data)[2]
            if f_score > best_f_score:
                best_f_score = f_score
                best_threshold = self.decision_threshold
            self.decision_threshold += steps
        self.decision_threshold = best_threshold
        print(f"Best Threshold: {self.decision_threshold}, F-Score: {best_f_score}")

    def evaluate(self, data):
        confusion_matrix = np.zeros([2, 2])
        for sentence in tqdm(data):
            prediction = self.predict(sentence)
            confusion_matrix[int(prediction), sentence.value] += 1
        if confusion_matrix.sum(1)[1] == 0:
            precision = 1
        else:
            precision = float(confusion_matrix[1, 1] / confusion_matrix.sum(1)[1])
        if confusion_matrix.sum(0)[1] == 0:
            recall = 1
        else:
            recall = float(confusion_matrix[1, 1] / confusion_matrix.sum(0)[1])
        if precision + recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)
        print(confusion_matrix)
        return precision, recall, f_score

    def predict(self, sentence):
        predicted_sense = self.best_fit(sentence)
        target_vector = self.embeddings.get_input_vector(sentence.target)
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
                except KeyError:
                    # print(f"Word {candidate} not in dictionary, ignoring candidate")
                    continue
            else:
                candidate_vector = self.embeddings.get_input_vector(candidate)
            similarity = Vectors.cos_sim(candidate_vector, context_vector)
            if similarity >= best_similarity:
                best_similarity = similarity
                best_candidate = candidate
        return best_candidate
