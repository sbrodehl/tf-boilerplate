from template import *
from template.misc import IteratorInitializerHook


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
        import importlib

        # enable tf logging, show DEBUG output
        tf.logging.set_verbosity(tf.logging.DEBUG)

        parser = argparse.ArgumentParser()
        parser.add_argument("input", type=str, help="input folder")
        parser.add_argument("--logdir", type=str, help="log folder")
        parser.add_argument("--dataset", type=str, default="data.mnist", help="dataset used")
        args = parser.parse_args()

        # import data
        dataset = importlib.import_module(args.dataset)
        sampler = dataset.DataSampler(args.input)

        NUMEXAMPLES = 60000
        BATCH_SIZE = 64
        EPOCHS = 1

        # define training dataset
        train_ds = sampler.training()
        train_ds = train_ds.cache().shuffle(buffer_size=NUMEXAMPLES).batch(BATCH_SIZE)
        train_ds = train_ds.repeat(EPOCHS)

        # define test dataset
        test_ds = sampler.testing()
        test_ds = test_ds.batch(1)

        # define validation dataset
        # val_ds = sampler.validation()

        # A reinitializable iterator is defined by its structure.
        # We could use the `output_types` and `output_shapes` properties of
        # either `training dataset` or `validation dataset` here,
        # because they are compatible (the same type and shape)
        iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                                   train_ds.output_shapes)
        data = iterator.get_next()

        # define all iterator initializer
        training_init_op = iterator.make_initializer(train_ds)
        testing_init_op = iterator.make_initializer(train_ds)  # here we reuse the train_ds

        with tf.name_scope("network"):
            net = network(*data)

        with tf.name_scope("loss"):
            loss = lossfn(net, *data)

        with tf.name_scope("train"):
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(net, 1), tf.cast(data[1], tf.int64))
                # define a summary for the accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', accuracy)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            do_train_batch = optimizer.minimize(loss, tf.train.get_or_create_global_step())

        #          #
        # TRAINING #
        #          #

        # define logging and saver
        log_steps = 2  # log every 2nd step
        save_mins = 5  # save every 5 min
        # evaluate logging directory
        logdir = args.logdir if args.logdir else "/tmp/example-logdir"
        # evaluate these tensors periodically
        logtensors = {
            "step": tf.train.get_or_create_global_step(),
            "accuracy": accuracy
        }

        # define all hooks
        hks = [
            # hook to save the summaries
            tf.train.SummarySaverHook(
                save_steps=log_steps,
                summary_op=tf.summary.merge_all(),
                output_dir=logdir
            ),
            # hook to save the model
            tf.train.CheckpointSaverHook(
                logdir,
                save_secs=60 * save_mins
            ),
            # hook to get logger output
            tf.train.LoggingTensorHook(
                logtensors,
                every_n_iter=log_steps
            ),
            # hook to initialize data iterators
            # iterator are initialized by placeholders
            # so we need to feed them during init
            IteratorInitializerHook(lambda s: s.run(
                training_init_op
            ))
        ]

        with tf.train.SingularMonitoredSession(
            hooks=hks,  # list of all hooks
            # checkpoint_dir=logdir  # restores checkpoint and continues training
        ) as sess:
            print(80 * '#')
            print('#' + 34 * ' ' + ' TRAINING ' + 34 * ' ' + '#')
            print(80 * '#')
            while not sess.should_stop():
                _ = sess.run(do_train_batch)

        #         #
        # TESTING #
        #         #

        # evaluate these tensors periodically
        logtensors = {
            "accuracy": accuracy
        }

        # define all hooks
        hks = [
            # hook to get logger output
            tf.train.LoggingTensorHook(
                logtensors,
                every_n_iter=1
            ),
            # hook to initialize data iterators
            # iterator are initialized by placeholders
            # so we need to feed them during init
            IteratorInitializerHook(lambda s: s.run(
                testing_init_op
            ))
        ]

        with tf.train.SingularMonitoredSession(
                hooks=hks,  # list of all hooks
                checkpoint_dir=logdir  # restores checkpoint
        ) as sess:
            print(80 * '#')
            print('#' + 34 * ' ' + ' TESTING ' + 35 * ' ' + '#')
            print(80 * '#')
            while not sess.should_stop():
                _ = sess.run(accuracy)

    # catch KeyboardInterrupt error message
    # IT WAS INTENTIONAL
    except KeyboardInterrupt:
        pass
