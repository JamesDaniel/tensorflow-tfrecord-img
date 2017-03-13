# REFERENCE https://gist.github.com/ambodi/408301bc5bc07bc5afa8748513ab9477
''''A module to read data'''

from PIL import Image
import numpy as np
import collections
from tensorflow.python.framework import dtypes
import tensorflow as tf

class DataSet(object):
    '''Dataset class object'''

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float64,
                 reshape=True):
        if reshape:
            assert images.shape[3] == 1
            images.reshape(images.shape[0],
            images.shape[1] * images.shape[2])

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epoch_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        '''Return the next 'batch_size' examples from this dataset.'''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, fake_data=False, one_hot=False,
                   dtype=dtypes.float64, reshape=True,
                   validation_size=763):
    '''set the images and labels'''
    num_training = 3053
    num_training = 5 # todo remove this line
    num_validation = 763
    num_validation = 5 # todo remove this line

    train_labels, train_images = _read_tfrecord('../out/train-00000-of-00001.tfrecords', num_training)
    validation_labels, validation_images = _read_tfrecord('../out/validation-00000-of-00001.tfrecords', num_validation)

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    #validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)

    #ds = collections.namedtuple('Datasets', ['train', 'validation'])

    return None#ds(train=train, validation=validation)

def _read_tfrecord(filename, num_examples):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    images = []
    labels = []

    for _ in range(num_examples):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64)
            }
        )
        img = features['image_raw']
        img = tf.image.decode_jpeg(img, channels=3)
        img.set_shape([640, 486, 3])
        label = tf.cast(features['label'], tf.int32)

        print(img)
        images.append(img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return labels, images
