import pandas as pd
import os
import numpy as np
from nltk.corpus import stopwords
import math
import time
from tqdm import tqdm
import re


class SWOWInterface:
    def __init__(self, number_of_responses, strength_file=None,use_ppmi=False):
        self.work_dir = "/projekte/semrel/WORK-AREA/Users/navid/SWOW-EN18"
        self.strength_file = strength_file
        self.response_file = "SWOW-EN.complete.20180827.csv"
        self.stops = stopwords.words("english")
        self.num_responses = number_of_responses
        self.use_ppmi=use_ppmi
        if self.strength_file:
            self.cues_to_responses, self.responses_to_cues = self.init_response_table()
        else:
            self.cues_to_responses, self.responses_to_cues, self.cue_response_count = (
                self.init_response_table()
            )
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

    def init_response_table(self):
        space_regex = re.compile(f"\\s")
        cues_to_responses = dict()
        responses_to_cues = dict()
        if not self.strength_file:
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
                if not self.strength_file:
                    if not cue in cue_response_amounts:
                        cue_response_amounts[cue] = {"#Total_Count#": 0}
                for response in responses:
                    if response in responses_to_cues:
                        responses_to_cues[response].add(cue)
                    else:
                        responses_to_cues[response] = set([cue])
                    if not self.strength_file:
                        cue_response_amounts[cue]["#Total_Count#"] += 1
                        if response in cue_response_amounts[cue]:
                            cue_response_amounts[cue][response] += 1
                        else:
                            cue_response_amounts[cue][response] = 1
        if not self.strength_file:
            return (cues_to_responses, responses_to_cues, cue_response_amounts)
        return (cues_to_responses, responses_to_cues)

    def calculate_strengths(self):
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
                denominator=0
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
        return pairs_to_strength, cues_to_index, responses

    def read_in_strengths(self):
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
        if not cue in self.cue_response_count:
            raise KeyError(f"{cue} not in dictionary")
        if not response in self.cue_response_count[cue]:
            raise KeyError(f"{response} not in dictionary")
        return (
            self.cue_response_count[cue][response]
            / self.cue_response_count[cue]["#Total_Count#"]
        )

    def init_strength_table(self):
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

    def get_candidate_set(self, cue):
        output = set()
        if cue in self.cues_to_responses:
            output.update(self.cues_to_responses[cue])
        if cue in self.responses_to_cues:
            output.update(self.responses_to_cues[cue])
        return output.difference(self.stops)

    def get_weighted_neighbours(self, token):
        neighbours = self.get_candidate_set(token)
        total_strength = np.array(
            [
                self.get_association_strength("star", candidate)
                for candidate in self.get_candidate_set("star")
            ]
        ).sum()
        return {
            neighbour: self.get_association_strength(token, neighbour) / total_strength
            for neighbour in neighbours
        }

    def get_association_strength(self, cue, response):
        strength = 0
        if (
            cue not in self.combined_cue_response_indices
            or response not in self.combined_cue_response_indices
        ):
            return 0
        cue_index = self.combined_cue_response_indices[cue]
        response_index = self.combined_cue_response_indices[response]
        strength += self.association_strength_matrix[cue_index, response_index]
        strength += self.association_strength_matrix[response_index, cue_index]
        return strength

    def get_association_strength_matrix(self, use_only_cues=True):
        if use_only_cues:
            num_cues = len(self.cue_indices)
            return (
                self.association_strength_matrix[:num_cues, :num_cues],
                self.cue_indices,
            )
        else:
            return self.association_strength_matrix, self.combined_cue_response_indices
