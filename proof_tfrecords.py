from PIL import Image
import numpy as np
import tensorflow as tf

def read_and_decode(filename_queue):
 reader = tf.TFRecordReader()
 _, serialized_example = reader.read(filename_queue)
 features = tf.parse_single_example(
  serialized_example,
  # Defaults are not specified since both keys are required.
  features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'depth': tf.FixedLenFeature([], tf.int64)
  })
 image = features['image_raw']
 label = tf.cast(features['label'], tf.int32)
 height = tf.cast(features['height'], tf.int32)
 width = tf.cast(features['width'], tf.int32)
 depth = tf.cast(features['depth'], tf.int32)
 return image, label, height, width, depth

def get_all_records(FILE):
 with tf.Session() as sess:
   filename_queue = tf.train.string_input_producer([ FILE ])
   image, label, height, width, depth = read_and_decode(filename_queue)
   image = tf.image.decode_jpeg(image, channels=3)

   init_op = tf.global_variables_initializer()
   sess.run(init_op)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)
   for i in range(3053):
     example, l = sess.run([image, label])
     img = Image.fromarray(example, 'RGB')
     img.save("output/" + str(i) + '-train.jpg')

     if not i % 100:
         print('images processed: ' + str(i))
     #print (example,l)
   coord.request_stop()
   coord.join(threads)

def main(_):
    filename_queue = get_all_records('out/train-00000-of-00001.tfrecords')


if __name__ == '__main__':
  tf.app.run(main=main)
