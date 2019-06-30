# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import cifar10
from datasets import imagenet

from datasets import pascalvoc_2007
from datasets import pascalvoc_2012

from datasets import dota
from datasets import drone_detection, drone_detection_to_tfrecords

from datasets import ucf101_24, 

datasets_map = {
    'cifar10': cifar10,
    'imagenet': imagenet,
    'pascalvoc_2007': pascalvoc_2007,
    'pascalvoc_2012': pascalvoc_2012,
    'dota': dota,
    'drone_detection': drone_detection,
    'ucf101_24': ucf101_24
}

datasets_map_tf = {
    'cifar10': cifar10,
    'imagenet': imagenet,
    'pascalvoc_2007': pascalvoc_2007,
    'pascalvoc_2012': pascalvoc_2012,
    'dota': dota,
    'drone_detection': drone_detection_to_tfrecords,
    'ucf101_24': ucf101_24_detection_to_tfrecords
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    """Given a dataset name and a split_name returns a Dataset.

    Args:
        name: String, the name of the dataset.
        split_name: A train/test split name.
        dataset_dir: The directory where the dataset files are stored.
        file_pattern: The file pattern to use for matching the dataset source files.
        reader: The subclass of tf.ReaderBase. If left as `None`, then the default
            reader defined by each dataset is used.
    Returns:
        A `Dataset` class.
    Raises:
        ValueError: If the dataset `name` is unknown.
    """
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_split(split_name,
                                        dataset_dir,
                                        file_pattern,
                                        reader)

def get_label_name_by_code(input_code, dataset_name):
    dataset_handler = datasets_map_tf[dataset_name]
    name = dataset_handler.LABEL_MAP.keys()
    code = dataset_handler.LABEL_MAP.values()

    reversed_label_map = dict(zip(code,name))
    print(reversed_label_map)
    return reversed_label_map[input_code]
