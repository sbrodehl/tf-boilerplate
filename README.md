# tf-boilerplate (tfbp)
![python-3 badge](https://img.shields.io/badge/python-3-brightgreen.svg) ![TensorFlow-2 badge](https://img.shields.io/badge/TensorFlow-2-brightgreen.svg)

TensorFlow boilerplate code using the [`tf.data` API](https://www.tensorflow.org/api_docs/python/tf/data) 
and the [`tf.train.MonitoredTrainingSession` API](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredSession) 
to build flexible and efficient input pipelines with simplified training 
in a distributed setting.

The modular structure allows you to replace the used network/model or dataset with a single argument,
and therefore makes it easy to compare various models, datasets and parameter settings.

## Getting Started

### Prerequisites

The current version requires in particular the following libraries / versions.

* [Python3](https://www.python.org/downloads/)
* [TensorFlow v2](https://github.com/tensorflow/tensorflow)

### Usage

#### Package Structure

The programm is split up into different (Python) packages, which are loaded by the main [tfbp.py](tfbp.py) just in time, just in sequence.

The following packages exist, and are required:

Package Path              | Class                                    | CLI argument(s)
------------------------- | ---------------------------------------- | ---------------
[data/](data)             | [BaseDataSampler](data/__init__.py)      | `-d/--data`
[experiment/](experiment) | [BaseExperiment](experiment/__init__.py) | `-e/--experiment`, `-o/--option`
[loss/](loss)             | [BaseLoss](loss/__init__.py)             | `-l/--loss`, `--loss-monitor` (optional)
[model/](model)           | [BaseModel](model/__init__.py)           | `-m/--model`

Each package itself can require additional arguments.

##### `data` Package

Currently, there are the following implementations of `BaseDataSampler`:

- [CIFAR10](data/cifar10/__init__.py)

- [CINIC10](data/cinic10/__init__.py)

- [MNIST](data/mnist/__init__.py)

- [Fashion MNIST](data/fashionmnist/__init__.py)

##### `experiment` Package

Currently, there are the following implementations of `BaseExperiment`:

- [Example](experiment/example.py)

##### `loss` Package

Currently, there are the following implementations of `BaseLoss`:

- [L2](loss/L2.py)

##### `model` Package

Currently, there are the following implementations of `BaseModel`:

- [cnn](model/cnn/__init__.py)

- [rcnn](model/rcnn/__init__.py)

#### Basic Usage

To run an experiment, the following CLI arguments are needed by the base classes (example output):

```bash
$ python3 tfbp.py

usage: tfbp [-h] -d {cifar10,cinic10,fashionmnist,mnist} -e {example} -o
            {training,testing,validation,debug} -l {l2}
            [--loss-monitor {l2} [{l2} ...]] -m {cnn,rcnn}
```

Set the required arguments to your desired values, and proceed:

```
$ python3 tfbp.py -d fashionmnist -l l2 -m rcnn -e example -o training
```

This loads the required packages and checks for additional, and required CLI arguments.
When all arguments are set, the experiment will be run.

In this case, the experiment does not do anything, only the dataset will be inistialized

```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 1s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
8192/5148 [===============================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 0s 0us/step

Process finished with exit code 0
```

Run `python3 tfbp.py --help` to see a complete list of command line arguments.

Currently we provide tf.data wrappers for [MNIST](http://yann.lecun.com/exdb/mnist/), [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html),
feel free to contribute others as well!

The RCNN and CNN models are simply for educational purpose.

## Features

Here is a short introduction to the used TensorFlow APIs.

For more information see the [references section](#references).

The code is structured very modular, all *models* and *datasets* are dynamically 
imported as modules, given the `--dataset` and `--model` argument.
Then, `tfbp.py` runs a basic training loop using the training dataset
to evaluate the `lossfn`, and minimizes the loss using the `AdamOptimizer`.
At the end, the model is evaluated using the testing dataset.

### tf.data API

To build flexible and efficient input pipelines we make use of the [`tf.data` API](https://www.tensorflow.org/api_docs/python/tf/data).

We introduce a simple `DataSampler` class, which has the abstract methods `training()`, `testing()` and `validation()`.
These methods must be implemented for each new dataset, and will be used during the training loop.

Any type of `tf.data.Dataset` can be returned, but e.g. `batch_size` and `epochs` will be set during a later stage by `tfbp.py`.
For an overview to the tf.data API see the [Importing Data Guide](https://www.tensorflow.org/guide/datasets).  

See the [MNIST example](data/mnist/__init__.py#L103).

### tf.train API

The tf.data API works well with the tf.train API for distributed execution, especially [`tf.train.MonitoredTrainingSession`](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredSession).
The class `MonitoredSession` provides a `tf.Session`-like object that handles initialization, recovery and hooks.

For distributed settings, use [`tf.train.MonitoredTrainingSession`](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredSession),
if not, [`tf.train.SingularMonitoredSession`](https://www.tensorflow.org/api_docs/python/tf/train/SingularMonitoredSession) is recommended.
For now, we use the class `SingularMonitoredSession`, as it provides all the goodies we need for the tf.data API.

If needed, the `SingularMonitoredSession` can be replaced with `MonitoredSession`.

Here is a basic example:
```python
# define a dataset
dataset = tf.data.Dataset(...).batch(32).repeat(5)
data = dataset.make_one_shot_iterator().get_next()

# define model, loss and optimizer
loss = network(data)
train_op = tf.train.AdamOptimizer().minimize(loss)

# SingularMonitoredSession example
# checkpoints and summaries are saved periodically 
saver_hook = CheckpointSaverHook(...)
summary_hook = SummarySaverHook(...)
with SingularMonitoredSession(hooks=[saver_hook, summary_hook]) as sess:
  while not sess.should_stop():
    sess.run(train_op)
```

Parameters like `batch_size` and `epoch` are implicit set via the Dataset.
Various hooks can be used to evaluate / process tensors during training, see [Training -> Training Hooks](https://www.tensorflow.org/api_guides/python/train#Training_Hooks)

For example
- `LoggingTensorHook` to log different tensors (e.g. current step, time or metrics)
- `CheckpointSaverHook` to save the model parameters
- `SummarySaverHook` to save summaries
- `OneTimeSummarySaverHook` to save summaries exactly once. (This can come handy to save the parameters of your run inside of your log and, thus, can be checked after training directly in tensorboard).

Logging the current step and accuracy, the command line output will look like (from the example above)
```
INFO:tensorflow:step = 70, accuracy = 0.90625 (0.006 sec)
```

For an overview see [Importing Data Guide - Using high-level APIs](https://www.tensorflow.org/guide/datasets#using_high_level_apis).

## References

Here is a (probably incomplete) list of resources, please contribute!

### tf.data & Datasets
- [tf.data: Fast, flexible, and easy-to-use input pipelines (TensorFlow Dev Summit 2018)](https://youtu.be/uIcqeP7MFH0) (youtu.be)
- [Google TF Datasets Intro](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html)
- [TF Importing Data](https://www.tensorflow.org/programmers_guide/datasets)
- [TF tf.data.Dataset docs](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

### ETL Input Pipeline
- [TF Performance Guides](https://www.tensorflow.org/performance/performance_guide)
- [TF Input Pipeline Performance Guide](https://www.tensorflow.org/performance/datasets_performance)

### Distributed Training
- [Distributed TensorFlow (TensorFlow Dev Summit 2018)](https://youtu.be/-h0cWBiQ8s8) (youtu.be)
- [Distributed TensorFlow training (Google I/O '18)](https://youtu.be/bRMGoPqsn20) (youtu.be)
- [TF tf.contrib.distribute](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute) (Docs)
- [TF tf.contrib.distribute.DistributionStrategy](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/DistributionStrategy) (Docs)
- [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed) (Docs)

### Misc
- [Training Performance: A user’s guide to converge faster (TensorFlow Dev Summit 2018)](https://youtu.be/SxOsJPaxHME) (youtu.be)
- [ResNet Model](https://github.com/tensorflow/models/tree/master/official/resnet) (TF Models Repo)
- [TF Performance Benchmarks](https://www.tensorflow.org/performance/benchmarks) (Docs)
- [TF Benchmarks](https://github.com/tensorflow/benchmarks) (Repo)
- [TF Testing Benchmarks](https://benchmarks-dot-tensorflow-testing.appspot.com) (Website)
- [@tfboyd/tf-tools](https://github.com/tfboyd/tf-tools)
- [@tfboyd/benchmark_harness](https://github.com/tfboyd/benchmark_harness)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.  
Feel free to open a PR or issue, we are happy to help!

## Versioning

We use [SemVer](http://semver.org/) for versioning.
For the versions available, see the [tags on this repository](https://github.com/sbrodehl/tf-boilerplate/tags). 

## Authors
* **Sebastian Brodehl** / [@sbrodehl](https://github.com/sbrodehl)
* **David Hartmann** / [@da-h](https://github.com/da-h)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgement
The dataset-handlers are taken from the [tensorflow/models](https://github.com/tensorflow/models) repository.
