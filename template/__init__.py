from pathlib import Path


class BaseDataSampler(object):
    """DataSampler which generates a TensorFlow Dataset object from given input.
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def training(self):
        raise NotImplementedError

    def testing(self):
        raise NotImplementedError

    def validation(self):
        raise NotImplementedError
