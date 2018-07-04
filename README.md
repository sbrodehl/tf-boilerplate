# **tf-boilerplate** -- normalized data-handling.
TensorFlow **boilerplate code** using **Dataset** and **MonitoredSession**  
![](https://img.shields.io/badge/Tensorflow-1.8-brightgreen.svg) ![](https://img.shields.io/badge/Python-3-brightgreen.svg)

## Features
With that it is easy to
- use the *native tensorflow-nodes* for **performant dataset-loading**
    - already included Datasets are: MNIST and CIFAR-10
- **monitor tensors** during training and testing (in *cli* and *tensorboard*)
- **modular datasets** and **models**, pass choice using cli
- save weights periodically -- restore at any point


## Getting Started

### Prerequisites

The current version requires in particular the following libraries / versions.

* [Python 3](https://www.python.org/downloads/), version `2.x` might work, no guarantees, no support.
* [TensorFlow v1.8](https://github.com/tensorflow/tensorflow) or newer.


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
The Dataset-handlers are taken from the [tensorflow/models](https://github.com/tensorflow/models)-Repository.
