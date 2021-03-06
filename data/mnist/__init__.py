from template import BaseDataSampler

import tensorflow as tf
from tensorflow import keras
import numpy as np


class DataSampler(BaseDataSampler):

    def __init__(self, data_dir):
        super().__init__(data_dir)

        mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        # normalize the values to [0,1] and of type float32
        train_images = train_images.astype(np.float32)
        test_images = test_images.astype(np.float32)
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        self.training_data = (train_images, train_labels.astype(np.int64))  # self.get_one_hot(train_labels, 10))
        self.testing_data = (test_images, test_labels.astype(np.int64))  # self.get_one_hot(test_labels, 10))

    @staticmethod
    def get_one_hot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return (res.reshape(list(targets.shape) + [nb_classes])).astype(np.int32)

    def training(self):
        """tf.data.Dataset object for the MNIST training data."""
        return tf.data.Dataset.from_tensor_slices(self.training_data)

    def testing(self):
        """tf.data.Dataset object for the MNIST test data."""
        return tf.data.Dataset.from_tensor_slices(self.testing_data)

    def validation(self):
        raise NotImplementedError("No validation data available for MNIST")
