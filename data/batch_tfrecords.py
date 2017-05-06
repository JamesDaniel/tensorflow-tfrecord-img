# REFERENCE https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32, """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('image_height', 486, """Provide square images of this size.""")

tf.app.flags.DEFINE_integer('image_width', 640, """Provide square images of this size.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 1,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 1.""")

tf.app.flags.DEFINE_integer('num_readers', 1, """Number of parallel readers during train.""")