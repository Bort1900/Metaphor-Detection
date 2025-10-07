import random
from data import Vectors
import numpy as np
import time
from tqdm import tqdm


class MaoModel:
    def __init__(self, dev_data, test_data, wn, embeddings, use_output_vec):
        self.dev_data = dev_data
        self.test_data = test_data
        self.wn = wn
        self.use_output = use_output_vec
        self.embeddings = embeddings
        self.decision_threshold = 0.6

    def train_threshold(self, increment, epochs, reduce_inc):
        for i in range(epochs):
            initial=time.time()
            print(f"epoch {i+1}:")
            correct_predictions = 0
            incorrect_predictions = 0
            random.shuffle(self.dev_data)
            for sentence in self.dev_data:
                moment = time.time()
                result = self.predict(sentence)
                print("predicion",time.time()-moment)
                if result > sentence.get_label():
                    self.decision_threshold -= increment
                    incorrect_predictions += 1
                elif result < sentence.get_label():
                    self.decision_threshold += increment
                    incorrect_predictions += 1
                else:
                    correct_predictions += 1
            print(
                "Accuracy: ",
                correct_predictions / (correct_predictions + incorrect_predictions),
            )
            if reduce_inc:
                increment /= 2
            print("epoch",time.time()-initial)

    def evaluate(self):
        confusion_matrix = np.zeros([2, 2])
        for sentence in tqdm(self.test_data):
            prediction = self.predict(sentence)
            confusion_matrix[int(prediction), sentence.get_label()] += 1
        precision = float(confusion_matrix[1, 1] / confusion_matrix.sum(1)[1])
        recall = float(confusion_matrix[1, 1] / confusion_matrix.sum(0)[1])
        f_score = 2 * precision * recall / (precision + recall)
        print(confusion_matrix)
        print(f"Precision: {precision}, Recall: {recall}, F-Score: {f_score}")

    def predict(self, sentence):
        predicted_sense = self.best_fit(sentence)
        target_vector = self.embeddings.get_input_vector(sentence.target)
        predicted_vector = self.embeddings.get_input_vector(predicted_sense)
        similarity = Vectors.cos_sim(target_vector, predicted_vector)
        return similarity < self.decision_threshold

    def best_fit(self, sentence):
        candidate_set = self.wn.get_candidate_set(sentence.target, "V")
        best_similarity = -1
        context_vector = self.embeddings.get_mean_vector(sentence.context)
        best_candidate = sentence.target
        for candidate in candidate_set:
            if self.use_output:
                try:
                    candidate_vector = self.embeddings.get_output_vector(candidate)
                except ValueError:
                    # print(f"Word {candidate} not in dictionary, ignoring candidate")
                    continue
            else:
                candidate_vector = self.embeddings.get_input_vector(candidate)
            similarity = Vectors.cos_sim(candidate_vector, context_vector)
            if similarity >= best_similarity:
                best_similarity = similarity
                best_candidate = candidate
        return best_candidate
