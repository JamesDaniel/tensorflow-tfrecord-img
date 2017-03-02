import numpy as np
import tensorflow as tf

def _separate_images_into_training_and_validation():
    pass

tf.app.flags.DEFINE_string('output_directory', 'out/',
                           'Output data directory')
tf.app.flags.DEFINE_string('validation_directory', 'images/validation/',
                           'Validation data directory')
tf.app.flags.DEFINE_integer('validation_shards', 2,
                            'Number of shards in validation TFRecord files.')
FLAGS = tf.app.flags.FLAGS


def _process_dataset(name, directory, num_shards, labels_file):
    print('_process_dataset')


def main(unused_argv):
    print('Saving records to %s' % FLAGS.output_directory)

if __name__ == '__main__':
    _process_dataset('validation', FLAGS.validation_directory + 'lights',
                     FLAGS.validation_shards, FLAGS.labels_file)
    tf.app.run()