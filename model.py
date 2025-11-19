import random
from data import Vectors
import numpy as np
import time
import torch
from nltk.corpus import stopwords
from torch.nn import CosineSimilarity
from tqdm import tqdm
from data import Sentence
import math
import matplotlib.pyplot as plt


class NThresholdModel:
    def __init__(
        self,
        dev_data,
        test_data,
        candidate_source,
        mean_multi_word,
        fit_embeddings,
        score_embeddings,
        use_output_vec,
        num_classes=2,
    ):
        """
        Model that categorizes Sentence data in n classes based on n-1 thresholds and uses a candidate set for getting the value
        dev_data: list of Sentence instances to train thresholds
        test_data: list of Sentence instances to evaluate model
        candidate_source: an object with a get_candidate_set function
        mean_multi_word: whether embeddings for multi-word tokens should be mean pooled from the embeddings of the individual words
        fit_embeddings: source for embeddings for finding best fit candidate
        score_embeddings: source for embeddings for scoring for prediction
        use_output_vec: whether ouput vectors(word2vec) should be used for comparing context to candidates
        num_classes: number of classes to classify
        """
        self.dev_data = dev_data
        self.test_data = test_data
        self.candidate_source = candidate_source
        self.mean_multi_word = mean_multi_word
        self.use_output = use_output_vec
        self.fit_embeddings = fit_embeddings
        self.score_embeddings = score_embeddings
        self.decision_thresholds = [0.5 for i in range(num_classes - 1)]
        self.num_classes = num_classes
        self.stops = stopwords.words("english")

    @staticmethod
    def calculate_scores(confusion_matrix):
        """
        calculates and returns precision, recall, and f-score metrics for a given evaluation as a dictionary
        confusion_matrix: result of the evaluation, prediction vs actual classes
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
        """
        returns a dictionary of recall, precision and f-score metrics after evaluating the model on test data
        data: test data for evaluation
        """
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
        """
        returns the class that the model predicts for a given instance
        sentence: instance to classify
        """
        try:
            similarity = self.get_compare_value(sentence)
        except ValueError:
            raise ValueError(f"{sentence.target} not in dictionary")
        scale = self.decision_thresholds + [similarity]
        scale.sort()
        return scale.index(similarity)

    def get_compare_value(self, sentence):
        """
        returns the value that is compared with the model's thresholds
        sentence: the instance that should be classified
        """
        predicted_sense = self.best_fit(sentence)
        try:
            target_vector = self.score_embeddings.get_input_vector(sentence.target)
            predicted_vector = self.score_embeddings.get_input_vector(predicted_sense)
        except KeyError:
            raise ValueError(f"{sentence.target} not in dictionary")
        return Vectors.cos_sim(target_vector, predicted_vector)

    def best_fit(self, sentence):
        """
        returns the best fiting instance from the candidate set to calculate the threshold
        sentence: the instance that should be classified
        """
        candidate_set = self.candidate_source.get_candidate_set(sentence.target)
        candidate_set.add(sentence.target_token)
        best_similarity = -1
        context = [word for word in sentence.context if word.lower() not in self.stops]
        context_vector = self.fit_embeddings.get_mean_vector(context)
        best_candidate = sentence.target
        for candidate in candidate_set:
            if self.use_output:
                try:
                    if len(candidate.split("_")) > 1 and self.mean_multi_word:
                        candidate_vector = self.fit_embeddings.get_mean_vector(
                            tokens=candidate.split("_"), use_input_vecs=False
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
                            tokens=candidate.split("_"), use_input_vecs=True
                        )
                    else:
                        candidate_vector = self.fit_embeddings.get_input_vector(
                            candidate
                        )
                except KeyError:
                    # print(f"Word {candidate} not in dictionary, ignoring candidate")
                    continue
            similarity = Vectors.cos_sim(candidate_vector, context_vector)

            if similarity >= best_similarity:
                best_similarity = similarity
                best_candidate = candidate
        return best_candidate

    def train_thresholds(self, increment, epochs):
        """
        trains the model's threshold on the dev_data
        increment: how much the threshold should be changed on a wrong prediction
        epochs: number of times the dev_data is run through the training process
        """
        data_per_class = [
            [sentence for sentence in self.dev_data if sentence.value == i]
            for i in range(self.num_classes)
        ]
        num_per_class = math.floor(len(self.dev_data) / self.num_classes)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            data = []
            for class_data in data_per_class:
                data += random.choices(population=class_data, k=num_per_class)
            random.shuffle(data)
            for sentence in tqdm(data):
                try:
                    comp_value = self.get_compare_value(sentence)
                    prediction = int(self.predict(sentence))
                except ValueError:
                    print(f"{sentence.target} not in dictionary, ignoring sentence")
                    continue
                if prediction != sentence.value:
                    for i, threshold in enumerate(self.decision_thresholds):
                        if comp_value > threshold and sentence.value <= i:
                            self.decision_thresholds[i] += increment
                        if comp_value < threshold and sentence.value > i:
                            self.decision_thresholds[i] -= increment
                self.decision_thresholds.sort()
            print(f"Current Thresholds: {self.decision_thresholds}")

    def evaluate_per_threshold(self, start, steps, increment, save_file):
        """
        writes some evaluation metrics into a file after evaluating the model with different thresholds
        start: the first threshold to test
        steps: the number of thresholds to test
        increment: the difference between two thresholds to test
        save_file: where to store the results
        """
        if self.num_classes > 2:
            raise ValueError("only works for 2 classes")
        self.decision_thresholds = [start]
        with open(save_file, "w", encoding="utf-8") as output:
            output.write(
                "Threshold\tPrecision\tRecall\tF1(Class 1)\tF1(Class 2)\tF1(Macro-Average)\n"
            )
            for i in range(steps):
                scores = self.evaluate(self.test_data)
                output.write(
                    f"{round(self.decision_thresholds[0],2)}\t{round(scores["precision_class_0"],2)}\t{round(scores["recall_class_0"],2)}\t{round(scores["f_1_class_0"],2)}\t{round(scores["f_1_class_1"],2)}\t{round(scores["macro_f_1"],2)}\n"
                )
                self.decision_thresholds[0] += increment

    def draw_distribution_per_class(self, save_file, labels, title):
        """
        draws box plots of the distributions of the prediction scores for each of the classes
        save_file: where the plots are stored
        """
        datapoints = [[] for _ in range(self.num_classes)]
        for sent in self.test_data:
            try:
                similarity = self.get_compare_value(sent)
            except ValueError:
                continue
            datapoints[sent.value].append(similarity)
        plt.boxplot(datapoints, labels=labels, orientation="horizontal")
        plt.title(title)
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()


class MaoModel(NThresholdModel):
    def __init__(
        self,
        dev_data,
        test_data,
        candidate_source,
        mean_multi_word,
        fit_embeddings,
        score_embeddings,
        use_output_vec,
    ):
        """
        Model that works like the model from the Mao(2018) paper, see NThresholdModel
        dev_data: list of Sentence instances to train thresholds
        test_data: list of Sentence instances to evaluate model
        candidate_source: an object with a get_candidate_set function
        mean_multi_word: whether embeddings for multi-word tokens should be mean pooled from the embeddings of the individual words
        fit_embeddings: source for embeddings for finding best fit candidate
        score_embeddings: source for embeddings for scoring for prediction
        use_output_vec: whether ouput vectors(word2vec) should be used for comparing context to candidates
        """
        super().__init__(
            dev_data=dev_data,
            test_data=test_data,
            candidate_source=candidate_source,
            mean_multi_word=mean_multi_word,
            fit_embeddings=fit_embeddings,
            score_embeddings=score_embeddings,
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
            batch_number = len(self.dev_data)
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
        """
        deprecated
        """
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
        """
        deprecated
        """
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


class ContextualMaoModel(NThresholdModel):
    def __init__(
        self,
        dev_data,
        test_data,
        candidate_source,
        mean_multi_word,
        fit_embeddings,
        score_embeddings,
        comparing_phrase,
        num_classes=2,
    ):
        """
        like Mao Model but uses contextual embeddings
        dev_data: list of Sentence instances to train thresholds
        test_data: list of Sentence instances to evaluate model
        candidate_source: an object with a get_candidate_set function
        mean_multi_word: whether embeddings for multi-word tokens should be mean pooled from the embeddings of the individual words
        embeddings: source for embeddings for comparing
        use_output_vec: whether ouput vectors(word2vec) should be used for comparing context to candidates
        comparing_phrase: Context to create candidate embeddings: '[Target] [comparing_phrase] [candidate]'
        num_classes: number of classes to classify
        """
        super().__init__(
            dev_data=dev_data,
            test_data=test_data,
            candidate_source=candidate_source,
            mean_multi_word=mean_multi_word,
            fit_embeddings=fit_embeddings,
            score_embeddings=score_embeddings,
            use_output_vec=False,
            num_classes=num_classes,
        )
        self.comparing_phrase = comparing_phrase
        self.cos = CosineSimilarity(dim=0, eps=1e-6)

    def best_fit(self, sentence):
        """
        returns the best candidate from the candidate set that fits into the sentence context
        sentence: sentence that will be predicted
        """
        candidate_set = self.candidate_source.get_candidate_set(sentence.target)
        candidate_set.add(sentence.target_token)
        best_similarity = -1
        context_vector = self.fit_embeddings.get_context_vector(sentence)
        best_candidate = sentence.target
        comparison_sentence = Sentence(
            sentence=f"{sentence.target} {self.comparing_phrase} xyz.",
            target="xyz",
            value=0,
        )
        for candidate in candidate_set:
            try:
                new_sent = comparison_sentence.replace_target(
                    candidate,
                    self.mean_multi_word,
                    target_index=comparison_sentence.target_index,
                )
            except:
                continue
            candidate_vector = self.fit_embeddings.get_input_vector(new_sent)
            similarity = self.cos(candidate_vector, context_vector)
            if similarity >= best_similarity:
                best_similarity = similarity
                best_candidate = candidate
        return best_candidate

    def get_compare_value(self, sentence):
        """
        returns the value that is used to determine the prediction
        sentence: Sentence instance that will be predicted
        """
        predicted_sense = self.best_fit(sentence)
        comparison_target_sentence = Sentence(
            sentence=f"{sentence.target} {self.comparing_phrase} {predicted_sense}.",
            target=sentence.target,
            value=0,
        )
        comparison_candidate_sentence = Sentence(
            sentence=f"{sentence.target} {self.comparing_phrase} xyz.",
            target="xyz",
            value=0,
        )
        comparison_candidate_sentence = comparison_candidate_sentence.replace_target(
            new_target=predicted_sense,
            split_multi_word=self.mean_multi_word,
            target_index=comparison_candidate_sentence.target_index,
        )
        target_vector = self.score_embeddings.get_input_vector(
            comparison_target_sentence
        )
        predicted_vector = self.score_embeddings.get_input_vector(
            comparison_candidate_sentence
        )
        return self.cos(target_vector, predicted_vector)


class ComparingModel(NThresholdModel):
    def __init__(
        self,
        dev_data,
        test_data,
        literal_embeddings,
        associative_embeddings,
        use_output_vec,
        num_classes=2,
    ):
        """
        model that compares literal and associative similarity and predicts metaphoricity with a threshold
        dev_data: list of Sentence instances to train thresholds
        test_data: list of Sentence instances to evaluate model
        literal_embeddings: Semantic Embeddings for comparing
        use_output_vec: whether ouput vectors(word2vec) should be used for comparing context to candidates
        num_classes: number of classes to classify
        associative_embeddings: WordAssociationEmbeddings instance
        """
        super().__init__(
            dev_data=dev_data,
            test_data=test_data,
            candidate_source=None,
            mean_multi_word=None,
            fit_embeddings=literal_embeddings,
            score_embeddings=associative_embeddings,
            use_output_vec=use_output_vec,
            num_classes=num_classes,
        )
        self.literal_embeddings = literal_embeddings
        self.associative_embeddings = associative_embeddings
        self.map_factor = 1  # mapping the size of one embedding space to the other for linear transform

    def get_compare_value(self, sentence):
        """
        get a value by comparing literal and associative similarities to context
        sentence: the Sentence instance for the calculation
        """
        try:
            literal_similarity, associative_similarity = self.get_similarities(sentence)
        except KeyError:
            raise ValueError("could not calculate comparison value")
        return self.map_factor * literal_similarity - associative_similarity

    def get_similarities(self, sentence):
        """
        returns the literal and associative similarity of the target to the context
        sentence: the Sentence instance for the calculation
        """
        try:
            literal_context_vec = self.literal_embeddings.get_mean_vector(
                sentence.context, not self.use_output
            )
            associative_context_vec = self.associative_embeddings.get_mean_vector(
                sentence.context
            )
            if self.use_output:
                literal_vec = self.literal_embeddings.get_output_vector(sentence.target)
            else:
                literal_vec = self.literal_embeddings.get_input_vector(sentence.target)
            associative_vec = self.associative_embeddings.get_input_vector(
                sentence.target
            )
        except KeyError:
            raise KeyError("Could not calculate the necessary embeddings")
        return Vectors.cos_sim(literal_context_vec, literal_vec), Vectors.cos_sim(
            associative_context_vec, associative_vec
        )

    def evaluate_per_threshold(self, start, steps, increment, save_file):
        """
        writes some evaluation metrics into a file after evaluating the model with different thresholds
        start: the first threshold to test
        steps: the number of thresholds to test
        increment: the difference between two thresholds to test
        save_file: where to store the results
        """
        self.estimate_map_factor()
        return super().evaluate_per_threshold(start, steps, increment, save_file)

    def estimate_map_factor(self):
        """
        estimates the factor for mapping from one embedding space to the other using dev data
        """
        # finding the ranges of the two embeddings spaces
        print("estimating mapping factor")
        smallest_literal = 10
        smallest_associative = 10
        largest_literal = -10
        largest_associative = -10
        ignore_count = 0
        for sentence in self.dev_data:
            try:
                literal_similarity, associative_similarity = self.get_similarities(
                    sentence
                )
            except KeyError:
                ignore_count += 1
                continue
            if literal_similarity < smallest_literal:
                smallest_literal = literal_similarity
            if literal_similarity > largest_literal:
                largest_literal = literal_similarity
            if associative_similarity < smallest_associative:
                smallest_associative = associative_similarity
            if associative_similarity > largest_associative:
                largest_associative = associative_similarity

        self.map_factor = (largest_associative - smallest_associative) / (
            largest_literal - smallest_literal
        )
        print(f"ignored {ignore_count} of {len(self.dev_data)} sentences")
        print(f"mapping factor: {self.map_factor}")

    def train_thresholds(self, increment, epochs):
        """
        trains the model's threshold on the dev_data
        increment: how much the threshold should be changed on a wrong prediction
        epochs: number of times the dev_data is run through the training process
        """
        ignore_count = 0
        self.estimate_map_factor()
        data_per_class = [
            [sentence for sentence in self.dev_data if sentence.value == i]
            for i in range(self.num_classes)
        ]
        num_per_class = math.floor(len(self.dev_data) / self.num_classes)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            data = []
            for class_data in data_per_class:
                data += random.choices(population=class_data, k=num_per_class)
            random.shuffle(data)
            for sentence in tqdm(data):
                try:
                    comp_value = self.get_compare_value(sentence)
                    prediction = int(self.predict(sentence))
                except ValueError:
                    ignore_count += 1
                    continue
                if prediction != sentence.value:
                    for i, threshold in enumerate(self.decision_thresholds):
                        if comp_value > threshold and sentence.value <= i:
                            self.decision_thresholds[i] += increment
                        if comp_value < threshold and sentence.value > i:
                            self.decision_thresholds[i] -= increment
                self.decision_thresholds.sort()
            print(f"ignored {ignore_count} of {len(data)}")
            print(f"Current Thresholds: {self.decision_thresholds}")


class RandomBaseline(MaoModel):
    def __init__(
        self,
        dev_data,
        test_data,
        candidate_source,
        score_embeddings,
    ):
        """
        Model that randomly choses a word from the candidate set to predict the class with a threshold
        dev_data: list of Sentence instances to train thresholds
        test_data: list of Sentence instances to evaluate model
        candidate_source: an object with a get_candidate_set function
        embeddings: source for embeddings for comparing

        """
        super().__init__(
            dev_data=dev_data,
            test_data=test_data,
            candidate_source=candidate_source,
            mean_multi_word=False,
            score_embeddings=score_embeddings,
            fit_embeddings=score_embeddings,
            use_output_vec=False,
        )

    def best_fit(self, sentence):
        """
        returns random element from the candidate set
        sentence: Sentence instance the model will predict
        """
        candidate_set = self.candidate_source.get_candidate_set(sentence.target)
        candidate_set.add(sentence.target_token)
        return random.choice(list(candidate_set))
