import fasttext


class FasttextModel:
    def __init__(self, load_file):
        self.model = fasttext.load_model(load_file)
        self.load_file = load_file

    def get_input_vector(self, token):
        return self.model.get_word_vector(token)

    def get_output_vector(self, token):
        return self.model.get_output_matrix()[self.model.get_word_id(token)]

    # def train(self, min_count=1):
    #     model = fasttext.train_unsupervised(input=self.data_file, model=self.model_type, min_count=min_count)
    #     model.save_model(self.save_file)
    #     self.model = model

    # @staticmethod
    # def loadModel(load_file, model_type, alternative_filename=None):
    #     save_file = load_file if not alternative_filename else alternative_filename
    #     load =
    #     model = FasttextModel(save_file, model_type)
    #     model.model = load
    #     return model
