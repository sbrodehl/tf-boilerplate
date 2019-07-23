from data import BaseDataSampler

import tensorflow as tf
from tensorflow import keras
import numpy as np


class FashionMNIST(BaseDataSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_data = None
        self.testing_data = None

    @staticmethod
    def get_one_hot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return (res.reshape(list(targets.shape) + [nb_classes])).astype(np.int32)

    def training(self):
        """tf.data.Dataset object for the Fashion MNIST training data."""
        return tf.data.Dataset.from_tensor_slices(self.training_data)

    def testing(self):
        """tf.data.Dataset object for the Fashion MNIST test data."""
        return tf.data.Dataset.from_tensor_slices(self.testing_data)

    def validation(self):
        raise NotImplementedError("No validation data available for Fashion MNIST")

    def setup(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        # normalize the values to [0,1] and of type float32
        train_images = train_images.astype(np.float32)
        test_images = test_images.astype(np.float32)
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        train_images = np.expand_dims(train_images, 3)
        test_images = np.expand_dims(test_images, 3)
        self.training_data = (train_images, train_labels.astype(np.int64))  # self.get_one_hot(train_labels, 10))
        self.testing_data = (test_images, test_labels.astype(np.int64))  # self.get_one_hot(test_labels, 10))

    def get_parameters(self):
        raise NotImplementedError

    def get_output_types(self) -> tuple:
        raise NotImplementedError

    def get_output_shapes(self) -> tuple:
        raise NotImplementedError

    def visualize(self, data, *args, **kwargs) -> bool:
        raise NotImplementedError
