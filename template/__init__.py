import zipfile
import random
import pickle
import bz2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path

from sklearn.model_selection import train_test_split
import multiprocessing
from multiprocessing import Pool

# TODO use TFRecords, e.g. see
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/how_tos/reading_data/convert_to_records.py
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/core/example/example.proto


class DataSampler(object):
    """DataSampler which generates a TensorFlow Dataset object from given 'OSG' input.
    """

    def __init__(self, data_dir, amount=30, threads=multiprocessing.cpu_count()):

        self.data_dir = Path(data_dir)

        self.train_amount = amount * 1.0 / 100.0
        self.threads = threads

        # empty datasets
        self.training_set = ([], [])
        self.testing_set = ([], [])
        self.validation_set = ([], [])

        self._prepare()

        print("####### OSG #######")
        print("Training    Set: {}".format(len(self.training_set[0])))
        print("Testing     Set: {}".format(len(self.testing_set[0])))
        print("Validation  Set: {}".format(len(self.validation_set[0])))
        print("##########  SUM: {}".format(
            len(self.validation_set[0])
            + len(self.testing_set[0])
            + len(self.training_set[0])
        ))

    def _prepare(self):
        # do some work

        # ...
        for ff in self.data_dir.glob("*.mhd"):
            print(ff)

        # X = [{"header": '1.mhd', "data": '1.raw', "size": (1, 2, 3), "UID": "123.234.5346546", "elsize": (0.1, 0.2, 0.3)}]
        X = ["1.mhd"]
        Y = [["LH-3", "LH-2"]]

        # do statistics

        self._split_train_test_val(X, Y)

    def _split_train_test_val(self, X, Y):
        # split up into train / test / val datasets
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=(1.0 - self.train_amount))

        self.validation_set = (x_val, y_val)   # 10 % for validation
        self.testing_set = (x_test, y_test)   # 0.9 * 0.3 = 27 % test
        self.training_set = (x_train, y_train)  # 0.9 * 0.6 = 64 % training

    def _build_dataset(self, set_):
        dataset = tf.data.Dataset.from_tensor_slices(set_)
        dataset = dataset.map(
            lambda x_obj, y_obj: tuple(tf.py_func(
                self._read_py_function, [x_obj, y_obj], [tf.float32, tf.int32, tf.float32]
            ))
        )
        # since the images have all different sizes we crop / pad them
        dataset = dataset.map(self._resize_function)
        return dataset

    def training(self):
        return len(self.training_set[0]), self._build_dataset(self.training_set)

    def testing(self):
        return len(self.testing_set[0]), self._build_dataset(self.testing_set)

    def validation(self):
        return len(self.validation_set[0]), self._build_dataset(self.validation_set)

    def _read_py_function(self, x_obj, y_obj):
        # convert to TFRecord?

        # map filename to data

        # x_obj = {"header": '1.mhd', "data": '1.raw', "size": (1, 2, 3), "UID": "123.234.5346546", "elsize": (0.1, 0.2, 0.3)}

        # read data from raw file
        data = np.zeros((), dtype=np.float32)
        label = np.zeros((), dtype=np.int32)
        meta = np.array([*x_obj["elsize"]], dtype=np.float32)
        return data, label, meta

    # Use standard TensorFlow operations to resize the image to a fixed shape.
    def _resize_function(self, data, label, meta):

        # normalize elsize to e.g. (1,1,1)

        # EXAMPLE
        # EXAMPLE
        # EXAMPLE

        # https://github.com/tensorflow/tensorflow/issues/521#issuecomment-182539003
        # cr.set_shape([None, None, None])
        # cr_resized = tf.image.resize_image_with_crop_or_pad(
        #     # yeah, img resize only works in 4D tensor...
        #     tf.expand_dims(cr, -1),
        #     self.mean_img_size[0], self.mean_img_size[1]
        # )
        # # apply fixed shape, since we know what we got
        # cr_resized.set_shape(
        #     [self.mean_img_size[0], self.mean_img_size[1], self.img_channels]
        # )
        # srt.set_shape([self.srt_dimension])
        # label.set_shape([self.classes])
        return data, label, meta


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input folder")
    args = parser.parse_args()

    sampler = DataSampler(args.input)

    _, train_ds = sampler.training()
    _, test_ds = sampler.testing()
    _, val_ds = sampler.validation()

    train_iter = train_ds.make_one_shot_iterator()
    test_iter = test_ds.make_one_shot_iterator()
    val_iter = val_ds.make_one_shot_iterator()

    train_batch = train_iter.get_next()
    test_batch = test_iter.get_next()
    val_batch = val_iter.get_next()

    with tf.train.SingularMonitoredSession() as sess:
        print(80 * '#')
        print(35 * '#' + ' TRAINING ' + 35 * '#')
        print(80 * '#')
        while not sess.should_stop():
            data, label, meta = sess.run(train_batch)
            print(label)

    with tf.train.SingularMonitoredSession() as sess:
        print(80 * '#')
        print(35 * '#' + ' TESTING ' + 36 * '#')
        print(80 * '#')
        while not sess.should_stop():
            data, label, meta = sess.run(test_batch)
            print(label)

    with tf.train.SingularMonitoredSession() as sess:
        print(80 * '#')
        print(34 * '#' + ' VALIDATION ' + 34 * '#')
        print(80 * '#')
        while not sess.should_stop():
            data, label, meta = sess.run(val_batch)
            print(label)
