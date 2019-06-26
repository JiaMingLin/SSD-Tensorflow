import os
import sys
import random
import pickle

import numpy as np
import tensorflow as tf

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'annotations/'
DIRECTORY_IMAGES = 'data/UCF101_24_Frame/Frames/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

LABEL_MAP = {
    'none': 0,
    'baseball':1, 'basketball':2, 'bike':3, 'climb_stair':4, 
             'eat':5, 'jump':6, 'raft':7, 'ride_motor':8, 'run':9, 
             'shake_hands':10, 'ski':11, 'surf':12, 'tennis':13, 'walk':14
}


def _process_image(directory, name):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    img_filename = os.join.path(directory, )



    filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    shape = [int(root.find('height').text),
             int(root.find('width').text),
             int(root.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(LABEL_MAP[label]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        ymin = max(float(bbox.find('ymin').text) / shape[0], 0)
        xmin = max(float(bbox.find('xmin').text) / shape[1], 0)
        ymax = min(float(bbox.find('ymax').text) / shape[0], 1.0)
        xmax = min(float(bbox.find('xmax').text) / shape[1], 1.0)
        """
        if (ymin < 0 or ymin > 1) or (xmin < 0 or xmin > 1) or (xmax < 0 or xmax > 1) or (ymax < 0 or ymax > 1):
            print(name)
            print("ymin: {}, xmin: {}, ymax: {}, xmax: {}".format(ymin, xmin, ymax, xmax))
        """
        bboxes.append((min(ymin,1.0),
                       min(xmin,1.0),
                       max(ymax,0),
                       max(xmax,0)
                       ))
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.
    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    print('dataset_dir: ', dataset_dir)
    print('{}.jpg'.format(name))
    #image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
    #    _process_image(dataset_dir, name)
    
    #example = _convert_to_example(image_data, labels, labels_text,
     #                             bboxes, shape, difficult)
    #tfrecord_writer.write(example.SerializeToString())

def _get_frames_filename(video_path):
    images_files = os.listdir(os.path.join(DIRECTORY_IMAGES, video_path))
    return [filename[:-4] for filename in images_files]


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='ucf_train', shuffling=False):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored. UCF101-24
      output_dir: Output directory.
    """

    # Dataset filenames, and shuffling.
    pickle_file = os.path.join(dataset_dir, 'UCF101-GT.pkl')
    
    with open(pickle_file, 'rb') as fid:
        cache = pickle.load(fid, encoding='latin1')
    
    video_list = cache['train_videos'][0]
    if name == 'ucf_test':
        video_list = cache['test_videos'][0]

    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(video_list)

    # Process dataset files.
    i = 0; j=0; k=0
    fidx = 0
    video_path = video_list[i]
    video_frame_list = _get_frames_filename(video_path)
    tf_filename = _get_output_filename(output_dir, name, fidx)
    while i < len(video_list):
        sys.stdout.write('\r>> Converting video %d/%d' % (i+1, len(video_list)))
        sys.stdout.flush()
        print(video_list[i])
        i += 1
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            nframes = len(video_frame_list)
            while j < SAMPLES_PER_FILES and k < nframes:

                frame_code = video_frame_list.pop()
                _add_to_tfrecord(dataset_dir, frame_code, tfrecord_writer)
                j += 1
                k += 1

            # finish on video, new one
            if len(video_frame_list) == 0:
                i += 1
                video_path = video_list[i]
                video_frame_list = _get_frames_filename(video_path)
            
            # if tfrecord is full
            if j == SAMPLES_PER_FILES:
                j = 0
                fidx += 1
                tf_filename = _get_output_filename(output_dir, name, fidx)
        #print("Converting image {}/{}".format(i, len(filenames)))

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the UCF101-24 dataset!')
