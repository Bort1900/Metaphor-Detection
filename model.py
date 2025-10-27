import random
from data import Vectors
import numpy as np
import time
from tqdm import tqdm
import math


class NThresholdModel:
    def __init__(
        self,
        dev_data,
        test_data,
        candidate_source,
        mean_multi_word,
        embeddings,
        use_output_vec,
        num_classes=2,
    ):
        self.dev_data = dev_data
        self.test_data = test_data
        self.candidate_source = candidate_source
        self.mean_multi_word = mean_multi_word
        self.use_output = use_output_vec
        self.embeddings = embeddings
        self.decision_thresholds = [0.5 for i in range(num_classes - 1)]
        self.num_classes = num_classes

    @staticmethod
    def calculate_scores(confusion_matrix):
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
            sum([scores[f"f_1_class_{i}"] for i in range(num_classes)]) / 2
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

    def evaluate(self, data):
        confusion_matrix = np.zeros([self.num_classes, self.num_classes])
        ignore_count = 0
        for sentence in tqdm(data):
            try:
                prediction = int(self.predict(sentence))
            except ValueError:
                # print(f"{sentence.target} not in dictionary, ignoring sentence")
                ignore_count += 1
                continue
            confusion_matrix[prediction, sentence.value] += 1
        print(confusion_matrix)
        print(f"ignored {ignore_count} sentences of {len(data)}")
        scores = NThresholdModel.calculate_scores(confusion_matrix=confusion_matrix)
        return scores

    def predict(self, sentence):
        try:
            similarity = self.get_compare_value(sentence)
        except ValueError:
            raise ValueError(f"{sentence.target} not in dictionary")
        scale = self.decision_thresholds + [similarity]
        scale.sort()
        return scale.index(similarity)

    def get_compare_value(self, sentence):
        predicted_sense = self.best_fit(sentence)
        try:
            target_vector = self.embeddings.get_input_vector(sentence.target)
        except KeyError:
            raise ValueError(f"{sentence.target} not in dictionary")
        predicted_vector = self.embeddings.get_input_vector(predicted_sense)
        return Vectors.cos_sim(target_vector, predicted_vector)

    def best_fit(self, sentence):
        candidate_set = self.candidate_source.get_candidate_set(sentence.target)
        candidate_set.add(sentence.target_token)
        best_similarity = -1
        context_vector = self.embeddings.get_mean_vector(sentence.context)
        best_candidate = sentence.target
        for candidate in candidate_set:
            if self.use_output:
                try:
                    if len(candidate.split("_")) > 1 and self.mean_multi_word:
                        candidate_vector = self.embeddings.get_mean_vector(
                            tokens=candidate.split("_"), use_input_vecs=False
                        )
                    else:
                        candidate_vector = self.embeddings.get_output_vector(candidate)
                except ValueError:
                    # print(f"Word {candidate} not in dictionary, ignoring candidate")
                    continue
            else:
                try:
                    if len(candidate.split("_")) > 1 and self.mean_multi_word:
                        candidate_vector = self.embeddings.get_mean_vector(
                            tokens=candidate.split("_"), use_input_vecs=True
                        )
                    else:
                        candidate_vector = self.embeddings.get_input_vector(candidate)
                except KeyError:
                    # print(f"Word {candidate} not in dictionary, ignoring candidate")
                    continue
            similarity = Vectors.cos_sim(candidate_vector, context_vector)

            if similarity >= best_similarity:
                best_similarity = similarity
                best_candidate = candidate
        return best_candidate

    def train_thresholds(self, increment, epochs):
        for _ in range(epochs):
            random.shuffle(self.dev_data)
            for sentence in self.dev_data:
                try:
                    comp_value = self.get_compare_value(sentence)
                    prediction = int(self.predict(sentence))
                except ValueError:
                    # print(f"{sentence.target} not in dictionary, ignoring sentence")
                    continue
                if prediction != sentence.value:
                    for i, threshold in enumerate(self.decision_thresholds):
                        if comp_value > threshold and sentence.value <= i:
                            self.decision_thresholds[i] += increment
                        if comp_value < threshold and sentence.value > i:
                            self.decision_thresholds[i] -= increment
                self.decision_thresholds.sort()


class MaoModel(NThresholdModel):
    def __init__(
        self,
        dev_data,
        test_data,
        candidate_source,
        mean_multi_word,
        embeddings,
        use_output_vec,
    ):
        super().__init__(
            dev_data=dev_data,
            test_data=test_data,
            candidate_source=candidate_source,
            mean_multi_word=mean_multi_word,
            embeddings=embeddings,
            use_output_vec=use_output_vec,
        )

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
                    f"Current_threshold: {self.decision_threshold}\nBatch F-score: {scores["macro_f_1"]}"
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
        last_f_score = self.evaluate(self.dev_data)[0]["macro_f_1"]
        while (
            self.decision_threshold <= upper_bound
            and self.decision_threshold >= lower_bound
            and i < max_epochs
        ):
            self.decision_threshold += direction * increment
            current_f_score = self.evaluate(self.dev_data)[0]["macro_f_1"]
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
            f_score = self.evaluate(self.dev_data)[0]["macro_f_1"]
            if f_score > best_f_score:
                best_f_score = f_score
                best_threshold = self.decision_threshold
            self.decision_threshold += steps
        self.decision_threshold = best_threshold
        print(f"Best Threshold: {self.decision_threshold}, F-Score: {best_f_score}")

    def evaluate_per_threshold(self, steps, save_file):
        self.decision_thresholds = [0]
        with open(save_file, "w", encoding="utf-8") as output:
            output.write(
                "Threshold\tPrecision\tRecall\tF1(Class 1)\tF1(Class 2)\tF1(Macro-Average)\n"
            )
            while self.decision_thresholds[0] < 1:
                scores = self.evaluate(self.test_data)
                output.write(
                    f"{round(self.decision_thresholds[0],2)}\t{round(scores["precision_class_0"],2)}\t{round(scores["recall_class_0"],2)}\t{round(scores["f_1_class_0"],2)}\t{round(scores["f_1_class_1"],2)}\t{round(scores["macro_f_1"],2)}\n"
                )
                self.decision_thresholds[0] += steps
