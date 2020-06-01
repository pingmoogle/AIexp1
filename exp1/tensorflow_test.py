# -*- coding: UTF-8 -*-

import tensorflow as tf

greeting = tf.constant('Hello Tensorflow!')
sess = tf.Session()
result = sess.run(greeting)
print(result)
sess.close()
