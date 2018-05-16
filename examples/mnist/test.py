import sys, os

sys.path.insert(0, os.path.abspath('.'))
import time
from tqdm import tqdm

from template import *
from download import dataset


class MNISTSampler(DataSampler):

    def _prepare(self):
        pass

    def training(self):
        """tf.data.Dataset object for MNIST training data."""
        return dataset(str(self.data_dir), 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')

    def testing(self):
        """tf.data.Dataset object for MNIST test data."""
        return dataset(str(self.data_dir), 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')


def network(data, labels_one_hot):
    data_format = 'channels_last'
    input_shape = [28, 28, 1]

    l = tf.keras.layers
    max_pool = l.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)
    # The model consists of a sequential chain of layers, so tf.keras.Sequential
    # (a subclass of tf.keras.Model) makes for a compact description.
    return tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape,
                input_shape=(28 * 28,)),
            l.Conv2D(
                32,
                5,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Conv2D(
                64,
                5,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Flatten(),
            l.Dense(1024, activation=tf.nn.relu),
            l.Dropout(0.4),
            l.Dense(10)
        ])(data)


def lossfn(net_out, data, labels_one_hot):
    with tf.name_scope('cross_entropy'):
        return tf.losses.sparse_softmax_cross_entropy(labels=labels_one_hot, logits=net_out)


if __name__ == '__main__':

    try:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("input", help="input folder")
        args = parser.parse_args()

        sampler = MNISTSampler(args.input)
        NUMEXAMPLES = 60000
        BATCH_SIZE = 64
        EPOCHS = 1

        # define training dataset
        train_ds = sampler.training()
        train_ds = train_ds.cache().shuffle(buffer_size=NUMEXAMPLES).batch(BATCH_SIZE)
        train_ds = train_ds.repeat(EPOCHS)
        train_iter = train_ds.make_one_shot_iterator()
        train_batch = train_iter.get_next()

        # define test dataset
        test_ds = sampler.testing()
        test_iter = test_ds.make_one_shot_iterator()
        test_batch = test_iter.get_next()

        # define validation dataset
        # val_ds = sampler.validation()
        # val_iter = val_ds.make_one_shot_iterator()
        # val_batch = val_iter.get_next()

        with tf.name_scope("network"):
            net = network(*train_batch)

        with tf.name_scope("loss"):
            loss = lossfn(net, *train_batch)

        with tf.name_scope("train"):
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(net, 1), tf.cast(train_batch[1], tf.int64))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            do_train_batch = optimizer.minimize(loss, tf.train.get_or_create_global_step())

        with tf.train.SingularMonitoredSession() as sess:
            print()
            print(80 * '#')
            print('#' + 34 * ' ' + ' TRAINING ' + 34 * ' ' + '#')
            print(80 * '#')
            accuracy_total = 0
            pbar = tqdm(total=NUMEXAMPLES / BATCH_SIZE * EPOCHS, desc="Training", leave=True)
            while not sess.should_stop():
                accuracy_res, _ = sess.run([accuracy, do_train_batch])
                accuracy_total += accuracy_res
                pbar.update(1)
                pbar.set_description('Accuracy %f' % accuracy_res)
                # print(sess.run(tf.train.get_or_create_global_step()))
                # print("Accuracy:",accuracy_res)
        accuracy_total /= pbar.n
        print()
        print("Total Train Accuracy:", accuracy_total)

        with tf.train.SingularMonitoredSession() as sess:
            print()
            print(80 * '#')
            print('#' + 34 * ' ' + ' TESTING ' + 35 * ' ' + '#')
            print(80 * '#')
            accuracy_total = 0
            pbar = tqdm(total=NUMEXAMPLES / BATCH_SIZE * EPOCHS, desc="Training", leave=True)
            while not sess.should_stop():
                accuracy_res, _ = sess.run([accuracy, do_train_batch])
                accuracy_total += accuracy_res
                pbar.update(1)
                pbar.set_description('Accuracy %f' % accuracy_res)
                # print(sess.run(tf.train.get_or_create_global_step()))
                # print("Accuracy:",accuracy_res)
        accuracy_total /= pbar.n
        print()
        print("Total Test Accuracy:", accuracy_total)

    except KeyboardInterrupt:
        pass
