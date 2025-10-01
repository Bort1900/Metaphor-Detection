import fasttext


class FasttextModel:
    def __init__(self, data_file, save_file, model_type):
        self.model = None
        self.save_file = save_file
        self.model_type = model_type
        self.data_file=data_file
        self.train()

    def train(self, min_count=1):
        model = fasttext.train_unsupervised(input=self.data_file, model=self.model_type, min_count=min_count)
        model.save_model(self.save_file)
        self.model = model

    @staticmethod
    def loadModel(load_file, model_type, alternative_filename=None):
        save_file = load_file if not alternative_filename else alternative_filename
        load = fasttext.load_model(load_file)
        model = FasttextModel(save_file, model_type)
        model.model = load
        return model
