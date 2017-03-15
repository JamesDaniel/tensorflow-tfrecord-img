#REFERENCE https://gist.github.com/ambodi/408301bc5bc07bc5afa8748513ab9477#file-dataset-py-L80
from __future__ import division
from __future__ import print_function

import argparse
import sys

import dataset

import tensorflow as tf

FLAGS = None

def main(_):
    '''Run the NN.'''
    print('starting app')
    mnist = dataset.read_data_sets(FLAGS.data_dir, one_hot=True)
    print('finished reading data sets')

    #x = tf.placeholder(tf.float32, [None, 640 * 486 * 3])
    #W = tf.Variable(tf.zeros([640 * 486 * 3, 53]))
    #b = tf.Variable(tf.zeros([53]))
    #y = tf.matmul(x, W) + b

    #y_ = tf.placeholder(tf.float32, [None, 53])

    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #sess = tf.InteractiveSession()
    #tf.global_variables_initializer.run()

    # Train
    #for _ in range(1000):
    #    batch_xs, batch_ys = mnist.train.next_batch(100)
    #    print(batch_xs.shape)
    #    print(batch_ys.shape)
    #    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(sess.run(accuracy, feed_dict={x: mnist.test.images,
    #                                    y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../out/train-00000-of-00001.tfrecords',
        help='Directory for storing data'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
