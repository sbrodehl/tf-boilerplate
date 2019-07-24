"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os
import tensorflow as tf
from data import BaseDataSampler
from .generate_cifar10_tfrecords import main as download

HEIGHT = 32
WIDTH = 32
DEPTH = 3
datashape = [HEIGHT, WIDTH, DEPTH]
numexamples = 50000


class Cifar10DataSet(object):
    """Cifar10 data set.

    Described by http://www.cs.toronto.edu/~kriz/cifar.html.
    """

    def __init__(self, data_dir, subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion

    def get_filenames(self):
        if self.subset == 'train':
            return [os.path.join(self.data_dir, 'train.tfrecords'), os.path.join(self.data_dir, 'validation.tfrecords')]

        if self.subset in ['train', 'validation', 'eval']:
            return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # Custom preprocessing.
        image = self.preprocess(image)

        return image, label

    def make_batch(self, batch_size=64):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames)  # .repeat()

        # Parse records.
        # dataset = dataset.map(self.parser, num_threads=batch_size, output_buffer_size=2 * batch_size)
        dataset = dataset.map(self.parser)

        # Potentially shuffle records.
        # if self.subset == 'train':
        #     min_queue_examples = int(Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
        #     Ensure that the capacity is sufficiently large to provide good random shuffling.
        #     dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

        return dataset

        # Batch it up.
        # dataset = dataset.batch(batch_size)
        # iterator = dataset.make_one_shot_iterator()
        # image_batch, label_batch = iterator.get_next()

        # return image_batch, label_batch

    def preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self.subset == 'train' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
            image = tf.image.random_flip_left_right(image)
        return image

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 45000
        elif subset == 'validation':
            return 5000
        elif subset == 'eval':
            return 10000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


class CIFAR10(BaseDataSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = None
        self.parser.add_argument("--datadir", type=str, default=self.data_dir,
                                 help="data training directory", required=True)

    def training(self):
        return Cifar10DataSet(self.data_dir, subset='train', use_distortion=False).make_batch()

    def testing(self):
        return Cifar10DataSet(self.data_dir, subset='eval', use_distortion=False).make_batch()

    def setup(self):

        # download and extract
        filepath = os.path.join(self.data_dir, "train.tfrecords")
        if not tf.gfile.Exists(self.data_dir):
            tf.gfile.MakeDirs(self.data_dir)
        if not tf.gfile.Exists(filepath):
            print("Downloading to " + filepath)
            print("Please wait...")
            download(self.data_dir)

    def validation(self):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    def get_output_types(self) -> tuple:
        raise NotImplementedError

    def get_output_shapes(self) -> tuple:
        raise NotImplementedError

    def visualize(self, data, *args, **kwargs) -> bool:
        raise NotImplementedError
