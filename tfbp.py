from template.misc import IteratorInitializerHook, OneTimeSummarySaverHook, ExperimentTemplate

import tensorflow as tf


if __name__ == '__main__':

    try:
        import argparse
        import importlib

        # enable tf logging, show DEBUG output
        tf.logging.set_verbosity(tf.logging.DEBUG)

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--datadir", type=str, default="/tmp/tf-boilerplate-data",
            help="data directory"
        )
        # model
        parser.add_argument(
            "--model", type=str, default="model.cnn",
            help="model used"
        )
        # training
        parser.add_argument(
            "--dataset", type=str, default="data.mnist",
            help="dataset used"
        )
        parser.add_argument(
            "--epoch", type=int, default=5,
            help="training epochs"
        )
        parser.add_argument(
            "--batchsize", type=int, default=64,
            help="batch size"
        )
        # logging / saving
        parser.add_argument(
            "--logdir", type=str, default="/tmp/tf-boilerplate-log",
            help="log folder"
        )
        parser.add_argument(
            "--save_mins", dest="save_mins", default=5, type=int,
            help="""Save the graph and summaries of once every N steps."""
        )
        parser.add_argument(
            "--log_steps", dest="log_steps", default=2, type=int,
            help="""Log the values of once every N steps."""
        )
        args = parser.parse_args()

        # add description text to your log
        tf.add_to_collection("SUMMARIES_ONCE", tf.summary.text("settings", tf.constant(
            ExperimentTemplate() + "\n".join(["| **" + i + "** | `" + str(k) + "` |" for i, k in vars(args).items()])
        ), collections="SUMMARIES_ONCE"))

        # import data
        dataset = importlib.import_module(args.dataset)
        sampler = dataset.DataSampler(args.datadir)

        # import model
        model = importlib.import_module(args.model)
        losses = importlib.import_module('.'.join([args.model, "loss"]))
        networks = importlib.import_module('.'.join([args.model, "network"]))
        # define network and loss function
        network = networks.network
        lossfn = losses.lossfn

        BATCH_SIZE = args.batchsize
        EPOCHS = args.epoch

        # define training dataset
        train_ds = sampler.training()
        train_ds = train_ds.batch(BATCH_SIZE)
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

        # evaluate these tensors periodically
        logtensors = {
            "step": tf.train.get_or_create_global_step(),
            "accuracy": accuracy
        }

        # define all hooks
        hks = [
            # hook to save the summaries
            tf.train.SummarySaverHook(
                save_steps=args.log_steps,
                summary_op=tf.summary.merge_all(),
                output_dir=args.logdir
            ),
            # hook to save the model
            tf.train.CheckpointSaverHook(
                args.logdir,
                save_secs=60 * args.save_mins
            ),
            # hook to get logger output
            tf.train.LoggingTensorHook(
                logtensors,
                every_n_iter=args.log_steps
            ),
            OneTimeSummarySaverHook(
                summary_op=tf.summary.merge_all("SUMMARIES_ONCE"),
                output_dir=args.logdir
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
            # checkpoint_dir=args.logdir  # restores checkpoint and continues training
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
                checkpoint_dir=args.logdir  # restores checkpoint
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
