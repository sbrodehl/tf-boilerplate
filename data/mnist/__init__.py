from .download import dataset
from template import BaseDataSampler


class DataSampler(BaseDataSampler):

    def _prepare(self):
        pass

    def training(self):
        """tf.data.Dataset object for MNIST training data."""
        return dataset(str(self.data_dir), 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')

    def testing(self):
        """tf.data.Dataset object for MNIST test data."""
        return dataset(str(self.data_dir), 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
