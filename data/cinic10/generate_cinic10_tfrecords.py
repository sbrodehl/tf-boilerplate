# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read CINIC-10 data from pickled numpy arrays and writes TFRecords.

Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CINIC-10 dataset downloaded from

Adapted for CINIC-10
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tarfile
import tensorflow as tf
import random
SEED = 42  # what else

CINIC_FILENAME = 'CINIC-10.tar.gz'
CINIC_DOWNLOAD_URL = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/' + CINIC_FILENAME
CINIC_LOCAL_FOLDER = 'cinic-10-batches-py'


def download_and_extract(data_dir):
    # download CINIC-10 if not already downloaded.
    tf.contrib.learn.datasets.base.maybe_download(CINIC_FILENAME, data_dir,
                                                  CINIC_DOWNLOAD_URL)
    tarfile.open(os.path.join(data_dir, CINIC_FILENAME), 'r:gz').extractall(data_dir)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


labels = {
    "airplane" : 0,  
    "automobile" : 1,
    "bird" : 2,
    "cat" : 3,
    "deer" : 4,
    "dog" : 5,
    "frog" : 6,
    "horse" : 7,
    "ship" : 8,
    "truck" : 9,
}


def _get_file_names(data_dir):
    file_names = {}
    for sub in ["train","valid","test"]:
        file_names[sub] = []
        for label in labels.keys():
            files = os.listdir(os.path.join(data_dir,sub,label))
            for f in files:
                file_names[sub].append(os.path.join(data_dir,sub,label,f))

    # to comply with rest of cifar-10 script
    file_names["validation"] = file_names["valid"]
    del file_names["valid"]
    file_names["eval"] = file_names["test"]
    del file_names["test"]

    return file_names


def label_from_filename(filename):
    label = filename.split("/")[-2]
    return labels[label]
    

def read_pickle_from_file(filename):
    with open(filename, 'rb') as f:
        data_dict = {
            "data" : f.read(),
            # "data" : f.tobytes(),
            # "data" : np.array(f).astype(np.uint8),
            "labels" : label_from_filename(filename)
        }
    return data_dict


def convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict['data']
            label = data_dict['labels']
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(data),  # .tobytes()),
                    'label': _int64_feature(label)
                }
            ))
            record_writer.write(example.SerializeToString())


def main(data_dir):
    print('Download from {} and extract.'.format(CINIC_DOWNLOAD_URL))
    download_and_extract(data_dir)
    file_names = _get_file_names(data_dir)
    input_dir = os.path.join(data_dir, CINIC_LOCAL_FOLDER)
    random.seed(SEED)
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        random.shuffle(input_files)
        output_file = os.path.join(data_dir, mode + '.tfrecords')
        print(mode, " contains ", len(input_files), " Files.")
        try:
            os.remove(output_file)
        except OSError:
            pass
        # Convert to tf.train.Example and write the to TFRecords.
        convert_to_tfrecord(input_files, output_file)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='',
        help='Directory to download and extract CINIC-10 to.')

    args = parser.parse_args()
    main(args.data_dir)
