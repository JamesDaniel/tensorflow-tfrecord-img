import numpy as np
import tensorflow as tf
import random
import os

tf.app.flags.DEFINE_string('train_directory', 'images/train',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', 'images/validation',
                           'Validation data directory')
tf.app.flags.DEFINE_string('labels_file', 'labels.txt', 'Labels file')
FLAGS = tf.app.flags.FLAGS


def _find_image_files(data_dir, labels_file):
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()]

    labels = []
    filenames = []
    texts = []

    label_index = 1

    for text in unique_labels:
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (
                label_index, len(labels)))
        label_index += 1

    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(unique_labels), data_dir))

    return filenames, texts, labels


def _move_some_imgs_to_validation(data_dir, labels_file):
    filenames, texts, labels = _find_image_files(data_dir, labels_file)
    num_of_pics = len(filenames)
    portion_of_pics = (num_of_pics / 100.)*20.
    portion_as_int = int(round(portion_of_pics))

    for i in xrange(portion_as_int):
        os.rename(filenames[i], filenames[i].replace('train', 'validation'))


def _make_validation_label_dirs(data_dir, labels_file):
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()]

    for i in xrange(len(unique_labels)):
        directory = data_dir + '/' + unique_labels[i]
        if not os.path.exists(directory):
            os.makedirs(directory)

    print('Finished making validation label dirs.')
    return True


def main(unused_argv):
    print('Finished running script ')

if __name__ == '__main__':
    _make_validation_label_dirs(FLAGS.validation_directory, FLAGS.labels_file)
    _move_some_imgs_to_validation(FLAGS.train_directory, FLAGS.labels_file)
    tf.app.run()
