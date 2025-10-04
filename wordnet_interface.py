import pandas as pd
import os


class WordNetInterface:
    def __init__(self):
        self.work_dir = "/resources/data/WordNet/WordNet-3.0_extract"
        self.token_to_synset_ids, self.synset_id_to_token, self.hypernym_synsets = (
            self.init_index_tables()
        )

    def init_index_tables(self):
        # token to synset tables
        nouns = pd.read_csv(
            os.path.join(self.work_dir, "noun-synsets.txt"),
            delimiter="\t",
            header=None,
            names=["token", "ids", "ambiguity"],
            dtype={"token": str, "ids": str, "ambiguity": int},
        )
        verbs = pd.read_csv(
            os.path.join(self.work_dir, "verb-synsets.txt"),
            delimiter="\t",
            header=None,
            names=["token", "ids", "ambiguity"],
            dtype={"token": str, "ids": str, "ambiguity": int},
        )
        nouns["POS"] = "NN"
        verbs["POS"] = "V"
        token_to_synset = pd.concat([nouns, verbs])
        token_to_synset["ids"] = token_to_synset["ids"].apply(
            lambda x: [int(id) for id in x.split(",")]
        )

        # synset to token tables
        syn_nouns = pd.read_csv(
            os.path.join(self.work_dir, "synset-nouns.txt"),
            delimiter="\t",
            header=None,
            names=["id", "tokens"],
            dtype={"id": int, "tokens": str},
        )
        syn_verbs = pd.read_csv(
            os.path.join(self.work_dir, "synset-verbs.txt"),
            delimiter="\t",
            header=None,
            names=["id", "tokens"],
            dtype={"id": int, "tokens": str},
        )
        syn_nouns["POS"] = "NN"
        syn_verbs["POS"] = "V"
        synset_to_token = pd.concat([syn_nouns, syn_verbs]).set_index("id")
        synset_to_token["tokens"] = synset_to_token["tokens"].apply(
            lambda x: [token.strip() for token in str(x).split(",")]
        )

        # hypernym tables
        noun_hypernyms = pd.read_csv(
            os.path.join(self.work_dir, "hyp_nouns.txt"),
            delimiter="\t",
            header=None,
            names=["hyponym", "hypernyms"],
            dtype={"hyponym": str, "hypernyms": str},
        )
        verb_hypernyms = pd.read_csv(
            os.path.join(self.work_dir, "hyp_verbs.txt"),
            delimiter="\t",
            header=None,
            names=["hyponym", "hypernyms"],
            dtype={"hyponym": str, "hypernyms": str},
        )
        noun_hypernyms["POS"] = "NN"
        verb_hypernyms["POS"] = "V"
        hypernyms = pd.concat([noun_hypernyms, verb_hypernyms])
        hypernyms["hypernyms"] = hypernyms["hypernyms"].apply(
            lambda x: [token.strip() for token in str(x).split(",")]
        )

        return token_to_synset, synset_to_token, hypernyms

    def get_synset_ids(self, token, pos):
        return self.concat_list_column(
            self.token_to_synset_ids[
                (self.token_to_synset_ids["token"] == token)
                & (self.token_to_synset_ids["POS"] == pos)
            ]["ids"]
        )

    def get_tokens_from_id(self, id):
        return self.synset_id_to_token.loc[id]["tokens"]

    def get_synonyms(self, token, pos):
        output = set()
        for id in self.get_synset_ids(token, pos):
            output.update(set(self.get_tokens_from_id(id)))
        return output

    def get_hypernyms(self, token, pos):
        return self.concat_list_column(
            self.hypernym_synsets[
                (self.hypernym_synsets["hyponym"] == token)
                & (self.hypernym_synsets["POS"] == pos)
            ]["hypernyms"]
        )

    def get_candidate_set(self, token, pos):
        candidates = set()
        candidates.update(self.get_synonyms(token, pos))
        candidates.update(set(self.get_hypernyms(token, pos)))
        return candidates

    @staticmethod
    def concat_list_column(col):
        return col.agg(lambda x: sum(x, []))
