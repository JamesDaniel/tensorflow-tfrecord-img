# REFERENCE http://learningtensorflow.com/lesson4/
import tensorflow as tf

x = tf.placeholder("float", [None, 3])
y = x * 2

with tf.Session() as session:
    x_data = [[1, 2, 3],
              [4, 5, 6]]
    result = session.run(y, feed_dict={x: x_data})
    print(result)
