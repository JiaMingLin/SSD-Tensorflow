import os
import sys
import random
import pickle

import numpy as np
import tensorflow as tf

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'UCF101-GT.pkl'
DIRECTORY_IMAGES = 'data/UCF101_24_Frame/Frames/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 1600

LABEL_MAP = {
    'None': 24, 'Basketball':0, 'BasketballDunk': 1, 'Biking': 2, 
    'CliffDiving':3, 'CricketBowling':4, 'Diving':5, 'Fencing':6, 
    'FloorGymnastics':7, 'GolfSwing':8, 'HorseRiding':9, 'IceDancing':10, 
    'LongJump':11, 'PoleVault':12, 'RopeClimbing':13, 'SalsaSpin':14, 
    'SkateBoarding':15, 'Skiing':16, 'Skijet':17, 'SoccerJuggling':18, 
    'Surfing':19, 'TennisSwing':20, 'TrampolineJumping':21, 'VolleyballSpiking':22, 'WalkingWithDog':23}


def _process_image(video_frame_dir, frame_code, CACHE):
    """Process a image and annotation file.

    Args:
      video_frame_dir: string, path to video frame directory e.g., 'Basketball/v_Basketball_g08_c02'.
      frame_code: the frame image file name
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    img_file = os.path.join(DIRECTORY_IMAGES, video_frame_dir, '{}.jpg'.format(frame_code))

    # TODO: image processing in tensorflow
    image_raw_data = tf.gfile.GFile(img_file, 'rb').read()

    resol = CACHE['resolution'][video_frame_dir]
    hight, width = resol[0], resol[1]
    shape = [hight, width, 3]
    # Read the annotations
    cate = video_frame_dir.split('/')[0]
    cate_code = LABEL_MAP[cate]
    action_tube_dict = CACHE['gttubes'][video_frame_dir]

    bboxes = []
    labels = []
    labels_text = []
    for action_code, tube_list in action_tube_dict.items():
        for tube in tube_list:
            for box in tube:
                frame_id = int(box[0])
                if frame_id == int(frame_code):
                    labels.append(action_code)
                    labels_text.append(cate.encode('ascii'))
                    ymin, xmin = box[1], box[2]
                    ymax, xmax = box[3], box[4]
                    ymin = max(ymin/hight, 0)
                    xmin = max(xmin / width, 0)
                    ymax = min(ymax / hight, 1.0)
                    xmax = min(xmax / width, 1.0)
                    bboxes.append((min(ymin,1.0),
                       min(xmin,1.0),
                       max(ymax,0),
                       max(xmax,0)
                       ))

    if len(labels) == 0:
        labels.append(LABEL_MAP['None'])
        labels_text.append('None'.encode('ascii'))
        bboxes.append((0,0,0,0))
        
    return image_raw_data, shape, bboxes, labels, labels_text


def _convert_to_example(image_data, labels, labels_text, bboxes, shape):
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
            #'image/object/bbox/difficult': int64_feature(difficult),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(video_frame_dir, frame_code, tfrecord_writer, CACHE):
    """Loads data from image and annotations files and add them to a TFRecord.
    Args:
      video_frame_ls: Dataset directory to video frames;
      frame_code: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text = _process_image(video_frame_dir, frame_code, CACHE)
    
    example = _convert_to_example(image_data, labels, labels_text, bboxes, shape)
    tfrecord_writer.write(example.SerializeToString())
    #tfrecord_writer.write('example'.encode('ascii'))

def _get_frames_filename(video_path):
    images_files = os.listdir(os.path.join(DIRECTORY_IMAGES, video_path))
    return [filename[:-4] for filename in images_files]


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='ucf101_24_detection_train', shuffling=False):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored. UCF101-24
      output_dir: Output directory.
    """

    # Dataset filenames, and shuffling.
    pickle_file = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    
    with open(pickle_file, 'rb') as fid:
        CACHE = pickle.load(fid, encoding='latin1')
    
    video_list = CACHE['train_videos'][0]
    if name == 'ucf101_24_detection_test':
        video_list = CACHE['test_videos'][0]

    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(video_list)

    # Process dataset files.
    i=0; j=0; k=0
    fidx = 0
    numvideo = len(video_list)
    video_path = video_list.pop()
    video_frame_list = _get_frames_filename(video_path)
    tf_filename = _get_output_filename(output_dir, name, fidx)
    total_frames = 0
    while len(video_list) > 0:
        sys.stdout.write('\r>> Converting video %d/%d' % (i+1, numvideo))
        sys.stdout.flush()
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            while j < SAMPLES_PER_FILES and len(video_frame_list) > 0:

                frame_code = video_frame_list.pop()
                _add_to_tfrecord(video_path, frame_code, tfrecord_writer, CACHE)
                j += 1
                total_frames += 1

            # finish on video, new one
            if len(video_frame_list) == 0:
                i += 1
                video_path = video_list.pop()
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
    print('Total frames: {}'.format(total_frames))
