from PIL import Image
import tensorflow as tf
from read_tfrecord import read_one_record
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_string('train_tfrecords', '../out/train-00000-of-00001.tfrecords',
                           'Training tfrecords file')
FLAGS = tf.app.flags.FLAGS

def _img_from_arr():
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([FLAGS.train_tfrecords])
        image, label, height, width, depth = read_one_record(filename_queue)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1):
            example, l = sess.run([image, label])
            img = Image.fromarray(example, 'RGB')

        coord.request_stop()
        coord.join(threads)
        return img

def main(_):
    raw_image_data = _img_from_arr()
    image = tf.placeholder("uint8", [None, None, 3])
    y = tf.image.rot90(image)




    with tf.Session() as session:
        result = session.run(y, feed_dict={image: raw_image_data})
        plt.imshow(result)
        plt.show()


if __name__ == '__main__':
    tf.app.run(main=main)
