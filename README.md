
# tf-boilerplate
![](https://img.shields.io/badge/Python-3-brightgreen.svg) ![](https://img.shields.io/badge/Tensorflow-1.8-brightgreen.svg)

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

###

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

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.  
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
The Dataset-handlers are taken from the [tensorflow/models](https://github.com/tensorflow/models) repository.
