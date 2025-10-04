import random
from data import Vectors
import numpy as np
import time


class MaoModel:
    def __init__(self, dev_data, test_data, wn, embeddings):
        self.dev_data = dev_data
        self.test_data = test_data
        self.wn = wn
        self.embeddings = embeddings
        self.decision_threshold = 0.6

    def train_threshold(self, increment, epochs):
        for i in range(epochs):
            correct_predictions = 0
            incorrect_predictions = 0
            random.shuffle(self.dev_data)
            for sentence in self.dev_data:
                result = self.predict(sentence)
                if result > sentence.get_label():
                    self.decision_threshold += increment
                    incorrect_predictions += 1
                elif result < sentence.get_label():
                    self.decision_threshold -= increment
                    incorrect_predictions += 1
                else:
                    correct_predictions += 1
                print(self.decision_threshold)
            print(
                "Accuracy: ",
                correct_predictions / (correct_predictions + incorrect_predictions),
            )

    def predict(self, sentence):
        pass

    def best_fit(self, sentence, use_output_vec):
        candidate_set = self.wn.get_candidate_set(sentence.target)
        best_similarity = -1
        context_vector = self.embeddings.get_mean_vector(sentence.context)
        best_candidate = sentence.target
        for candidate in candidate_set:
            print(candidate)
            if use_output_vec:
                try:
                    candidate_vector = self.embeddings.get_output_vector(candidate)
                except ValueError:
                    print(f"Word {candidate} not in dictionary, ignoring candidate")
                    continue
            else:
                candidate_vector = self.embeddings.get_input_vector(candidate)
            similarity = Vectors.cos_sim(candidate_vector, context_vector)
            print(similarity)
            if similarity >= best_similarity:
                best_similarity = similarity
                best_candidate = candidate
        return best_candidate
