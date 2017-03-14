# REFERENCE http://learningtensorflow.com/lesson2/
import numpy as np
data = np.random.randint(1000, size=10000)

import tensorflow as tf


x = tf.constant(data, name='x')
y = tf.Variable((5 * (x**2) - (3 * x) + 15), name='y')


model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
