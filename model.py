import random
import numpy as np


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

    def best_fit(self, sentence):
        candidate_set = self.wn.get_candidate_set(sentence.target)
        context_embeddings = [
            self.embeddings.get_input_vector(token) for token in sentence.context
        ]
        context_vector = np.mean(context_embeddings, axis=0)
