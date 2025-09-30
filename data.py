import wordnet
from nltk.tokenize import word_tokenize
class TargetWord:
    def __init__(self,token):
        self.token = token
        self.wn = wordnet.WordNetInterface()
    def get_candidate_set(self):
        candidates=set()
        candidates.update(self.wn.get_synonyms(self.token))
        candidates.update(set(self.wn.get_hypernyms(self.token)))
        return candidates

class Sentence:
    def __init__(self,sentence,target):
        self.tokens = word_tokenize(sentence)
        self.target = TargetWord(target)
        target_index = self.tokens.index(target)
        self.context = self.tokens[:target_index]+self.tokens[target_index+1:]
    