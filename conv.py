#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import code

def buildDQN(action_num=4):
  inpt = tf.placeholder(tf.float32, shape=(10,84,84,4), name="input_layer")
  conv1 = tf.layers.conv2d(inputs=inpt, filters=16, kernel_size=[8,8], strides=4, activation=tf.nn.relu)
  conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4,4], strides=2, activation=tf.nn.relu)
  conv2_r = tf.reshape(conv2, [-1, 9*9*32])
  dense = tf.layers.dense(inputs=conv2_r, units=256, activation=tf.nn.relu)
  
  linear_W = tf.Variable(tf.truncated_normal([256, action_num]))
  outpt = tf.matmul(dense, linear_W)
  return outpt, inpt


ins = np.random.rand(10,84,84,4)
network, inpt = buildDQN()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

result = sess.run(network, feed_dict={inpt: ins})
print(result)
code.interact(local=locals())