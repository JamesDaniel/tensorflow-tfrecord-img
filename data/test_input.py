# REFERENCE http://stackoverflow.com/questions/42509811/gentlely-way-to-read-tfrecords-data-into-batches
'''
    This script will attempt to read in a tfrecords file and feed it directly
    into a neural network. This is different from using a dataset to get the
    next batch.
'''
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

tf.app.flags.DEFINE_integer('target_image_height', 150, 'train input image height')
tf.app.flags.DEFINE_integer('target_image_width', 200, 'train input image width')

tf.app.flags.DEFINE_integer('batch_size', 12, 'batch size of training.')
tf.app.flags.DEFINE_integer('num_epochs', 100, 'epochs of training.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate of training.')

FLAGS = tf.app.flags.FLAGS


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
            # 'image/height': tf.FixedLenFeature([], tf.int64),
            # 'image/width': tf.FixedLenFeature([], tf.int64),
            # 'image/channels': tf.FixedLenFeature([], tf.int64),
            # 'image/encoded': tf.FixedLenFeature([], tf.string),
            # 'image/class/label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], out_type=tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channels = tf.cast(features['depth'], dtype=tf.int32)
    label = tf.cast(features['label'], dtype=tf.int32)

    # cast image int64 to float32 [0, 255] -> [-0.5, 0.5]
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image_shape = tf.stack([640, 486, 3])
    image = tf.reshape(image, image_shape)

    return image, label


def inputs(train, batch_size, num_epochs):
    if not num_epochs:
        num_epochs = None
    filenames = ['../out/train-00000-of-00001.tfrecords']
    #print(filenames)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    image, label = read_and_decode(filename_queue)
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=2,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)

    return images, sparse_labels


def run_training():
    images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)

    #images = tf.Print(images, [images], message='this is images:')
    #predictions = inference.lenet(images=images, num_classes=5, activation_fn='relu')
    #slim.losses.softmax_cross_entropy(predictions, labels)

    #total_loss = slim.losses.get_total_loss()
    #tf.summary.scalar('loss', total_loss)

    #optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)

    #train_op = slim.learning.create_train_op(total_loss=total_loss,
    #                                         optimizer=optimizer,
    #                                         summarize_gradients=True)
    #slim.learning.train(train_op=train_op, save_summaries_secs=20)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
