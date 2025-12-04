import pandas as pd
import os
import numpy as np
from nltk.corpus import stopwords
import math
import time
from tqdm import tqdm
import re


class SWOWInterface:
    def __init__(
        self,
        number_of_responses,
        strength_file=None,
        use_ppmi=False,
        candidate_cap=0,
    ):
        self.work_dir = "/projekte/semrel/WORK-AREA/Users/navid/SWOW-EN18"
        self.strength_file = strength_file
        self.response_file = "SWOW-EN.complete.20180827.csv"
        self.stops = stopwords.words("english")
        self.num_responses = number_of_responses
        self.use_ppmi = use_ppmi
        self.cues_to_responses, self.responses_to_cues, self.cue_response_count = (
            self.init_response_table()
        )
        self.most_frequent_pos = self.read_in_pos_freq()
        """
        Interface for the Small World of Words Word Association Data by Simon De Deyne that can give out candidate sets for words
        the data consists of cues where people have given three responses they associate with the cue word
        number_of_responses: the number of responses (max 3) to take into account
        strength_file: if exists a tsv file with the columns cue, response, number of mentions, total number of responses to the cue, association strength, if None it will be generated
        use_ppmi: whether to use pointwise mutual information as a measure for the association strength when there is no strength file
        candidate_cap: number of occurences a candidate needs to have for it to appear in the candidate set
        """
        (
            self.association_strength_matrix,
            self.cue_indices,
            self.response_indices,
        ) = self.init_strength_table()

        self.combined_cue_response_indices = {
            token: indices[token]
            for indices in [self.cue_indices, self.response_indices]
            for token in indices
        }
        self.candidate_cap = candidate_cap

    def init_response_table(self):
        """
        returns three dictionaries
        cues_to_responses: a list of responses for every cue
        responses_to_cues: a list of cues for every response
        cue_response_count: a dict of number of occurence for every cue and response
        """
        space_regex = re.compile(f"\\s")
        cues_to_responses = dict()
        responses_to_cues = dict()
        cue_response_amounts = dict()
        with open(
            os.path.join(self.work_dir, self.response_file), "r", encoding="utf-8"
        ) as assocs:
            assocs.readline()
            for line in assocs:
                values = [item.strip()[1:-1] for item in line.split(",")]
                if len(values) != 18:
                    continue
                cue = re.sub(space_regex, "_", values[11])
                responses = [
                    re.sub(space_regex, "_", values[15 + i])
                    for i in range(self.num_responses)
                    if values[15 + i] != "No more responses"
                    and values[15 + i] != "Unknown word"
                ]
                if len(responses) == 0:
                    continue
                if cue in cues_to_responses:
                    cues_to_responses[cue].update(responses)
                else:
                    cues_to_responses[cue] = set(responses)
                if not cue in cue_response_amounts:
                    cue_response_amounts[cue] = {"#Total_Count#": 0}
                for response in responses:
                    if response in responses_to_cues:
                        responses_to_cues[response].add(cue)
                    else:
                        responses_to_cues[response] = set([cue])
                    cue_response_amounts[cue]["#Total_Count#"] += 1
                    if response in cue_response_amounts[cue]:
                        cue_response_amounts[cue][response] += 1
                    else:
                        cue_response_amounts[cue][response] = 1
        return (cues_to_responses, responses_to_cues, cue_response_amounts)

    def calculate_strengths(self):
        """
        returns the association strengths of cue response pairs as a dict, the cue indices as a cue,index dict and set of responses that are not cues
        """
        cues_to_index = dict()
        responses = set()
        pairs_to_strength = dict()
        num_cues = len(self.cues_to_responses)
        for i, cue in enumerate(self.cue_response_count):
            cues_to_index[cue] = i
            for response in self.cue_response_count[cue]:
                if response == "#Total_Count#":
                    continue
                responses.add(response)
                denominator = 0
                if self.use_ppmi:
                    for cue_i in self.responses_to_cues[response]:
                        denominator += self.get_relative_probability(cue_i, response)
                    pairs_to_strength[(cue, response)] = max(
                        0,
                        math.log2(
                            self.get_relative_probability(cue, response)
                            * num_cues
                            / denominator
                        ),
                    )
                else:
                    pairs_to_strength[(cue, response)] = self.get_relative_probability(
                        cue, response
                    )
        for cue in self.cue_response_count:
            total = 0
            for response in self.cue_response_count[cue]:
                if (cue, response) in pairs_to_strength:
                    total += pairs_to_strength[(cue, response)]
            for response in self.cue_response_count[cue]:
                if (cue, response) in pairs_to_strength:
                    pairs_to_strength[(cue, response)] /= total

        return pairs_to_strength, cues_to_index, responses

    def read_in_pos_freq(self):
        """
        reads in and returns the most frequent part of speech for a token
        """
        most_frequent = dict()
        with open("most_frequent_pos.tsv", "r", encoding="utf-8") as freqs:
            for line in freqs:
                pair = [word.strip() for word in line.split()]
                most_frequent[pair[0]] = pair[1]
        return most_frequent

    def get_pos(self, token):
        """
        returns most frequent part of speech for a token according to frequency file, defaults to noun
        """
        tags = {"NN": "n", "JJ": "a", "VB": "v", "NI": "n", "NP": "n", "NR": "n"}
        token = token.split("_")[0]
        if token in self.most_frequent_pos:
            tag = self.most_frequent_pos[token]
        else:
            return "n"
        if tag in tags:
            return tags[tag]
        else:
            return tag

    def read_in_strengths(self):
        """
        reads in and returns from self.strength_file the association strengths of cue response pairs as a dict, the cue indices as a cue,index dict and set of responses that are not cues

        """
        space_regex = re.compile(f"\\s")
        pairs_to_strength = dict()
        cue_to_index = dict()
        responses = set()
        with open(
            os.path.join(self.work_dir, self.strength_file), "r", encoding="utf-8"
        ) as strengths:
            strengths.readline()
            for line in strengths:
                values = [value.strip() for value in line.split("\t")]
                if len(values) != 5:
                    continue
                cue = re.sub(space_regex, "_", values[0])
                response = re.sub(space_regex, "_", values[1])
                strength = float(values[4])
                if (cue, response) in pairs_to_strength:
                    pairs_to_strength[(cue, response)] += strength
                else:
                    pairs_to_strength[(cue, response)] = strength
                if cue not in cue_to_index:
                    cue_to_index[cue] = len(cue_to_index)
                responses.add(response)
        return pairs_to_strength, cue_to_index, responses

    def write_strengths_to_file(self, filepath):
        """
        writes the strengths to a file in the strenght file tsv format cue,response,response count, cue count, strength
        """
        with open(filepath, "w", encoding="utf-8") as output:
            output.write(
                "cue"
                + "\t"
                + "response"
                + "\t"
                + "R_Count"
                + "\t"
                + "N"
                + "\t"
                + "Strength"
                + "\n"
            )
            for cue in self.cue_response_count:
                for response in self.cue_response_count[cue]:
                    if response == "#Total_Count#":
                        continue
                    cue_index = self.combined_cue_response_indices[cue]
                    response_index = self.combined_cue_response_indices[response]
                    output.write(
                        str(cue)
                        + "\t"
                        + str(response)
                        + "\t"
                        + str(self.cue_response_count[cue][response])
                        + "\t"
                        + str(self.cue_response_count[cue]["#Total_Count#"])
                        + "\t"
                        + str(
                            self.association_strength_matrix[cue_index, response_index]
                        )
                        + "\n"
                    )

    def get_relative_probability(self, cue, response):
        """
        returns the relative probability that the response has been given to this cue
        """
        if not cue in self.cue_response_count:
            raise KeyError(f"{cue} not in dictionary")
        if not response in self.cue_response_count[cue]:
            raise KeyError(f"{response} not in dictionary")
        return (
            self.cue_response_count[cue][response]
            / self.cue_response_count[cue]["#Total_Count#"]
        )

    def init_strength_table(self):
        """
        returns the association strength matrix and the cue and response indices for navigating the matrix
        """
        if not self.strength_file:
            pairs_to_strength, cue_to_index, responses = self.calculate_strengths()
        else:
            pairs_to_strength, cue_to_index, responses = self.read_in_strengths()
        responses.difference_update(cue_to_index.keys())
        num_cues = len(cue_to_index)
        response_to_index = {
            response: i + num_cues for i, response in enumerate(responses)
        }
        num_responses = len(response_to_index)
        association_strength_matrix = np.zeros(
            [num_cues + num_responses, num_cues + num_responses]
        )
        for tuple in pairs_to_strength:
            cue = tuple[0]
            response = tuple[1]
            cue_index = cue_to_index[cue]
            if response in cue_to_index:
                response_index = cue_to_index[response]
            else:
                response_index = response_to_index[response]
            association_strength_matrix[cue_index, response_index] = pairs_to_strength[
                tuple
            ]
        return association_strength_matrix, cue_to_index, response_to_index

    def get_num_occurrences(self, word_1, word_2):
        """
        returns the number of times one of the two words has been given as a response to the other one
        """
        total = 0
        if (
            word_1 in self.cue_response_count
            and word_2 in self.cue_response_count[word_1]
        ):
            total += self.cue_response_count[word_1][word_2]
        if (
            word_2 in self.cue_response_count
            and word_1 in self.cue_response_count[word_2]
        ):
            total += self.cue_response_count[word_2][word_1]
        return total

    def get_candidate_set(self, word, pos=None):
        """
        returns all the responses given to the word and all the cues the word was given as a response to
        pos: restricts result to a certain part of speech if given(v:verb,n:noun,a:adjective)
        """
        output = set()
        if word in self.cues_to_responses:
            output.update(
                [
                    response
                    for response in self.cues_to_responses[word]
                    if self.get_num_occurrences(word, response) >= self.candidate_cap
                    if not pos or pos == self.get_pos(response)
                ]
            )
        if word in self.responses_to_cues:
            output.update(
                [
                    cue
                    for cue in self.responses_to_cues[word]
                    if self.get_num_occurrences(cue, word) >= self.candidate_cap
                    if not pos or pos == self.get_pos(cue)
                ]
            )
        return output.difference(self.stops)

    def get_weighted_neighbours(self, token):
        """
        returns a dictionary that gives the association strength for all neighbours of the token in the word association graph
        """
        neighbours = self.get_candidate_set(token)
        total_strength = np.array(
            [
                self.get_association_strength(token, candidate)
                for candidate in self.get_candidate_set(token)
            ]
        ).sum()
        return {
            neighbour: self.get_association_strength(token, neighbour) / total_strength
            for neighbour in neighbours
        }

    def get_association_strength(self, word_1, word_2):
        """
        returns the association strength of the two words in the bidirectional word association graph
        """
        strength = 0
        if (
            word_1 not in self.combined_cue_response_indices
            or word_2 not in self.combined_cue_response_indices
        ):
            return 0
        cue_index = self.combined_cue_response_indices[word_1]
        response_index = self.combined_cue_response_indices[word_2]
        strength += self.association_strength_matrix[cue_index, response_index]
        strength += self.association_strength_matrix[response_index, cue_index]
        return strength

    def get_association_strength_matrix(self, use_only_cues=True):
        """
        returns the association strength matrix and the indices to navigate the matrix
        use_only_cues: if True only the cues will be in the CxC matrix, if False the responses will also be in the (C+R)x(C+R) matrix
        """
        if use_only_cues:
            num_cues = len(self.cue_indices)
            return (
                self.association_strength_matrix[:num_cues, :num_cues],
                self.cue_indices,
            )
        else:
            return self.association_strength_matrix, self.combined_cue_response_indices
