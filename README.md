# tf-boilerplate
![python-3 badge](https://img.shields.io/badge/python-3-brightgreen.svg) ![TensorFlow-1.8 badge](https://img.shields.io/badge/TensorFlow-1.8-brightgreen.svg)

TensorFlow boilerplate code using the [tf.data API](https://youtu.be/uIcqeP7MFH0) 
and the [tf.train.MonitoredTrainingSession API](https://youtu.be/-h0cWBiQ8s8) 
to build flexible and efficient input pipelines with simplified training 
in a distributed setting.

The modular structure allows you to replace the used network/model or dataset with a single argument,
and therefore makes it easy to compare various models, datasets and parameter settings.

## Getting Started

### Prerequisites

The current version requires in particular the following libraries / versions.

* [Python 3](https://www.python.org/downloads/)
* [TensorFlow v1.8](https://github.com/tensorflow/tensorflow)

### Usage

To run a simple CNN on MNIST use the following command

```bash
python3 tfbp.py --dataset data.mnist --model model.cnn
```

which produces the following output

```
$ python3 tfbp.py --dataset data.mnist --model model.cnn
Downloading https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz to /tmp/tmpsw_g1ad9.gz
Downloading https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz to /tmp/tmpwrbbj9qq.gz
Downloading https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz to /tmp/tmp6w2dezra.gz
Downloading https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz to /tmp/tmpm_pr5i0j.gz
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
################################################################################
#                                   TRAINING                                   #
################################################################################
INFO:tensorflow:Saving checkpoints for 1 into /tmp/tf-boilerplate-log/model.ckpt.
INFO:tensorflow:step = 0, accuracy = 0.125
INFO:tensorflow:step = 10, accuracy = 0.09375 (0.006 sec)
INFO:tensorflow:step = 20, accuracy = 0.5625 (0.006 sec)
INFO:tensorflow:step = 30, accuracy = 0.78125 (0.006 sec)
INFO:tensorflow:step = 40, accuracy = 0.828125 (0.005 sec)
INFO:tensorflow:step = 50, accuracy = 0.9375 (0.005 sec)
INFO:tensorflow:step = 60, accuracy = 0.921875 (0.005 sec)
INFO:tensorflow:step = 70, accuracy = 0.90625 (0.006 sec)
INFO:tensorflow:step = 80, accuracy = 0.9375 (0.005 sec)
INFO:tensorflow:step = 90, accuracy = 0.890625 (0.007 sec)
INFO:tensorflow:step = 100, accuracy = 0.953125 (0.005 sec)
...
```

Run `python3 tfbp.py --help` to see a complete list of command line arguments.

Currently we provide tf.data wrappers for [MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), feel free to contribute others as well!

The CNN model is simply for educational purpose.

## Features

Here is a short introduction to the used TensorFlow APIs.

For more information see the [references section](#references).

### tf.data API

### tf.train.MonitoredTrainingSession API

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
