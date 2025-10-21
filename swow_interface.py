import pandas as pd
import os
import numpy as np
from nltk.corpus import stopwords
import time
from tqdm import tqdm
import re


class SWOWInterface:
    def __init__(self):
        self.work_dir = "/projekte/semrel/WORK-AREA/Users/navid/SWOW-EN18"
        self.association_table, self.association_strength_table = self.init_tables()
        self.stops = stopwords.words("english")
        self.index()

    def init_tables(self):
        associations = pd.read_csv(
            os.path.join(self.work_dir, "SWOW-EN.complete.20180827.csv"),
            usecols=["id", "cue", "R1Raw", "R2Raw", "R3Raw", "R1", "R2", "R3"],
        )
        associations[["cue", "R1", "R2", "R3"]] = associations[
            ["cue", "R1", "R2", "R3"]
        ].replace(to_replace=re.compile(f"\\s"), value="_")
        strengths = pd.read_csv(
            os.path.join(self.work_dir, "strength.SWOW-EN.R123.20180827.csv"),
            keep_default_na=False,
            delimiter="\t",
        )
        strengths[["cue", "response"]] = strengths[["cue", "response"]].replace(
            to_replace=re.compile(f"\\s"), value="_"
        )
        return associations, strengths

    def index(self):
        cue_arr = self.association_strength_table["cue"].to_numpy()
        response_arr = self.association_strength_table["response"].to_numpy()
        cues, cue_indices, cue_counts = np.unique(
            cue_arr, return_index=True, return_counts=True
        )
        self.cue_to_index = {
            cue: (index, index + count)
            for cue, index, count in zip(cues, cue_indices, cue_counts)
        }
        self.response_to_index = dict()
        for i, response in enumerate(response_arr):
            if response in self.response_to_index:
                self.response_to_index[response].append(i)
            else:
                self.response_to_index[response] = [i]

    def get_candidate_set(self, cue):
        cue_table = self.association_table[self.association_table["cue"] == cue][
            ["R1", "R2", "R3"]
        ]
        associations = []
        for col in cue_table.columns:
            associations += [assoc for assoc in cue_table[col]]
        return set(associations).difference(self.stops)

    def get_neighbour_nodes(self, token):
        output = set()
        output.update(
            [
                cue
                for cue in self.association_strength_table[
                    self.association_strength_table["response"] == token
                ]["cue"]
            ]
        )
        output.update(
            [
                response
                for response in self.association_strength_table[
                    self.association_strength_table["cue"] == token
                ]["response"]
            ]
        )
        return output

    def get_weighted_neighbours(self, token):
        df = self.association_strength_table
        cue_arr = df["cue"].to_numpy()
        resp_arr = df["response"].to_numpy()
        strength_arr = df["R123.Strength"].to_numpy()
        cue_index = (0, 0)
        response_index = []
        if token in self.cue_to_index:
            cue_index = self.cue_to_index[token]
        if token in self.response_to_index:
            response_index = self.response_to_index[token]
        if not cue_index and not response_index:
            return dict()
        tokens = np.concatenate(
            [
                resp_arr[cue_index[0] : cue_index[1]],
                cue_arr[response_index],
            ]
        )
        strengths = np.concatenate(
            [
                strength_arr[cue_index[0] : cue_index[1]],
                strength_arr[response_index],
            ]
        )
        try:
            unique, inverse = np.unique(tokens, return_inverse=True)
        except TypeError as er:
            print([token for token in tokens if type(token) != str])
            raise TypeError(er.message)
        sums = np.bincount(inverse, weights=strengths)
        return dict(zip(unique, sums))

    def get_association_strength(self, cue, response):
        value = list(
            self.association_strength_table[
                self.association_strength_table["cue"] == cue
            ][self.association_strength_table["response"] == response]["R123.Strength"]
        )
        if len(value) == 0:
            return 0
        return value[0]

    def get_association_strength_matrix(self, use_only_cues=True):
        cues = self.association_strength_table["cue"].unique()
        num_cues = len(cues)
        cue_indices = {cue: i for i, cue in enumerate(cues)}
        if use_only_cues:
            valid = self.association_strength_table[
                self.association_strength_table["response"].isin(cues)
            ]
        else:
            responses = self.association_strength_table["response"].unique()
            for response in responses:
                if response not in cue_indices:
                    cue_indices[response] = num_cues
                    num_cues += 1
            valid = self.association_strength_table
        association_matrix = np.zeros([len(cue_indices), len(cue_indices)])
        row_indices = valid["cue"].map(cue_indices)
        col_indices = valid["response"].map(cue_indices)
        association_matrix[row_indices, col_indices] = valid["R123.Strength"]
        return association_matrix, cue_indices
