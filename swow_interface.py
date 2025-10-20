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
            delimiter="\t",
        )
        strengths[["cue", "response"]] = strengths[["cue", "response"]].replace(
            to_replace=re.compile(f"\\s"), value="_"
        )
        return associations, strengths

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
        output = dict()
        responses = self.association_strength_table[
            self.association_strength_table["cue"] == token
        ]
        for i in range(len(responses)):
            line = responses.iloc[i]
            output[line["response"]] = line["R123.Strength"]
        cues = self.association_strength_table[
            self.association_strength_table["response"] == token
        ]
        for i in range(len(cues)):
            line = cues.iloc[i]
            if line["cue"] in output:
                output[line["cue"]] += line["R123.Strength"]
            else:
                output[line["cue"]] = line["R123.Strength"]
        return output

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
