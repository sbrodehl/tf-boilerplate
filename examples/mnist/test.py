import sys, os
sys.path.insert(0, os.path.abspath('.'))

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


def network(data):
    data_format = 'channels_last'
    if data_format == 'channels_first':
        input_shape = [1, 28, 28]
    else:
      assert data_format == 'channels_last'
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
              ])(data[0])
    # return data[0]

def loss(batch,logits):
    return tf.losses.sparse_softmax_cross_entropy(labels=batch[1], logits=logits)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input folder")
    args = parser.parse_args()

    sampler = MNISTSampler(args.input)

    train_ds = sampler.training()
    train_ds = train_ds.cache().shuffle(buffer_size=50000).batch(64)
    train_ds = train_ds.repeat(5)
    test_ds = sampler.testing()
    # val_ds = sampler.validation()

    train_iter = train_ds.make_one_shot_iterator()
    test_iter = test_ds.make_one_shot_iterator()
    # val_iter = val_ds.make_one_shot_iterator()

    train_batch = train_iter.get_next()
    test_batch = test_iter.get_next()
    # val_batch = val_iter.get_next()

    net = network(train_batch)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    do_train_batch = optimizer.minimize(loss(train_batch,net), tf.train.get_or_create_global_step())
    # est = tf.estimator.EstimatorSpec(
    #     mode=tf.estimator.ModeKeys.TRAIN,
    #     loss=loss,
    #     train_op=do_train_batch
    # mnist_classifier = tf.estimator.Estimator(
    #   model_fn=est,
    #   # model_dir=flags_obj.model_dir,
    #   # params={
    #       # 'data_format': data_format,
    #       # 'multi_gpu': flags_obj.multi_gpu
    #  # }
    # )

    with tf.train.SingularMonitoredSession() as sess:
        print(80 * '#')
        print(35 * '#' + ' TRAINING ' + 35 * '#')
        print(80 * '#')
        while not sess.should_stop():
            data, label, meta = sess.run(do_train_batch)
            print(label)

    # with tf.train.SingularMonitoredSession() as sess:
    #     print(80 * '#')
    #     print(35 * '#' + ' TESTING ' + 36 * '#')
    #     print(80 * '#')
    #     while not sess.should_stop():
    #         data, label, meta = sess.run(test_batch)
    #         print(label)

