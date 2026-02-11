import os
import numpy as np
from nltk.corpus import stopwords
import math
from nltk.stem import WordNetLemmatizer
import re
from wordnet_interface import CandidateSource


class SWOWInterface(CandidateSource):
    def __init__(
        self,
        number_of_responses: int,
        response_file: str,
        strength_file: str | None = None,
        use_ppmi: bool = False,
        candidate_cap: int = 0,
    ):
        """
        Interface for the Small World of Words Word Association Data by Simon De Deyne that can give out candidate sets for words
        the data consists of cues where people have given three responses they associate with the cue word
        :param number_of_responses: the number of responses (max 3) to take into account
        :param response_file: filepath to the file listing the three responses for all cues and all participants
        :param strength_file: if exists a tsv file with the columns cue, response, number of mentions, total number of responses to the cue, association strength, if None it will be generated
        :param use_ppmi: whether to use pointwise mutual information as a measure for the association strength when there is no strength file
        :param candidate_cap: number of occurences a candidate needs to have for it to appear in the candidate set
        """
        self.strength_file = strength_file
        self.response_file = response_file
        self.stops = stopwords.words("english")
        self.num_responses = number_of_responses
        self.use_ppmi = use_ppmi
        self.cue_response_count, self.responses_to_cues = self.init_response_table()
        self.most_frequent_pos = self.read_in_pos_freq()
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
        self.wnl = WordNetLemmatizer()

    def init_response_table(
        self,
    ) -> tuple[dict[str, dict[str, int]], dict[str, set[str]]]:
        """
        returns two dictionaries
        cue_response_count: a dict of number of occurences for every cue and response
        responses_to_cues: a list of cues for every response
        """
        space_regex = re.compile(f"\\s")
        responses_to_cues: dict[str, set[str]] = dict()
        cue_response_amounts = dict()
        with open(self.response_file, "r", encoding="utf-8") as assocs:
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
        return (cue_response_amounts, responses_to_cues)

    def calculate_strengths(
        self,
    ) -> tuple[dict[tuple[str, str], float], dict[str, int], list[str]]:
        """
        returns the association strengths of cue response pairs as a dict, the cue indices as a cue,index dict and a list of responses that are not cues
        """
        cues_to_index = dict()
        responses = set()
        pairs_to_strength = dict()
        num_cues = len(self.cue_response_count)
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

        responses.difference_update(cues_to_index.keys())
        return pairs_to_strength, cues_to_index, list(responses)

    def read_in_pos_freq(self) -> dict[str, str]:
        """
        reads in and returns the most frequent part of speech for a token
        """
        most_frequent = dict()
        with open("most_frequent_pos.tsv", "r", encoding="utf-8") as freqs:
            for line in freqs:
                pair = [word.strip() for word in line.split()]
                most_frequent[pair[0]] = pair[1]
        return most_frequent

    def get_pos(self, token: str) -> str:
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

    def read_in_strengths(
        self,
    ) -> tuple[dict[tuple[str, str], float], dict[str, int], list[str]]:
        """
        reads in and returns from self.strength_file the association strengths of cue response pairs as a dict, the cue indices as a cue,index dict and set of responses that are not cues

        """
        space_regex = re.compile(f"\\s")
        pairs_to_strength = dict()
        cue_to_index = dict()
        responses = set()
        if not self.strength_file:
            raise ValueError("strength file cannot be none")
        with open(self.strength_file, "r", encoding="utf-8") as strengths:
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

        responses.difference_update(cue_to_index.keys())
        return pairs_to_strength, cue_to_index, list(responses)

    def write_strengths_to_file(self, filepath: str):
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

    def get_relative_probability(self, cue: str, response: str) -> float:
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

    def init_strength_table(self) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
        """
        returns the association strength matrix and the cue and response indices for navigating the matrix
        """
        if not self.strength_file:
            pairs_to_strength, cue_to_index, responses = self.calculate_strengths()
        else:
            pairs_to_strength, cue_to_index, responses = self.read_in_strengths()
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

    def get_num_occurrences(self, word_1: str, word_2: str) -> int:
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

    def get_neighbours(self, word: str, directional: bool = False) -> list[str]:
        """
        returns all the neighbours to the word in the word association graph
        :param directional: whether the neighbours that are cues to the word should be given in addition to the responses, True: only responses
        """
        output = []
        if word in self.cue_response_count:
            output += [
                response
                for response in self.cue_response_count[word]
                if self.get_num_occurrences(word, response) >= self.candidate_cap
                and response != "#Total_Count#"
            ]

        if word in self.responses_to_cues and not directional:
            output += [
                cue
                for cue in self.responses_to_cues[word]
                if self.get_num_occurrences(cue, word) >= self.candidate_cap
            ]

        return output

    def get_candidate_set(self, token: str, pos: list[str] | None = None) -> set[str]:
        """
        returns all the responses given to the token and all the cues the token was given as a response to as well as the token itself
        :param pos: list of parts of speech, restricts result to a certain part of speech if given(v:verb,n:noun,a:adjective)
        """
        output = set(
            [
                token
                for token in self.get_neighbours(token, directional=False)
                if not pos or self.get_pos(token) in pos
            ]
        )
        output.add(token)
        return output.difference(self.stops)

    def get_weighted_neighbours(self, token: str) -> dict[str, float]:
        """
        returns a dictionary that gives the association strength for all neighbours of the token in the word association graph
        """
        neighbours = self.get_neighbours(token, directional=False)
        total_strength = np.array(
            [
                self.get_association_strength(token, candidate)
                for candidate in neighbours
            ]
        ).sum()
        if total_strength == 0 and len(neighbours) != 0:
            breakpoint()
        return {
            neighbour: self.get_association_strength(token, neighbour) / total_strength
            for neighbour in neighbours
        }

    def get_association_strength(
        self, word_1: str, word_2: str, directional: bool = False
    ) -> float:
        """
        returns the association strength of the two words  word association graph

        :param directional: whether both direction's weights should be added, True: just one direction
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
        if not directional:
            strength += self.association_strength_matrix[response_index, cue_index]
        return strength

    def get_association_strength_matrix(
        self, use_only_cues: bool = True
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        returns the association strength matrix and the indices to navigate the matrix
        :param use_only_cues: if True only the cues will be in the CxC matrix, if False the responses will also be in the (C+R)x(C+R) matrix
        """
        if use_only_cues:
            num_cues = len(self.cue_indices)
            return (
                self.association_strength_matrix[:num_cues, :num_cues],
                self.cue_indices,
            )
        else:
            return self.association_strength_matrix, self.combined_cue_response_indices

    def get_nodes(self, only_cues: bool = False) -> list[str]:
        """
        Returns all the nodes in the word association graph

        :param only_cues: Whether only cues or also responses should be given out
        """
        if only_cues:
            return [cue for cue in self.cue_indices]
        else:
            return [token for token in self.combined_cue_response_indices]
