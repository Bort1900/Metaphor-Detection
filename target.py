import wordnet
class TargetWord:
    def __init__(self,token):
        self.token = token
        self.wn = wordnet.WordNetInterface()
    def get_candidate_set(self):
        candidates=set()
        candidates.update(self.wn.get_synonyms(self.token))
        candidates.update(set(self.wn.get_hypernyms(self.token)))
        return candidates