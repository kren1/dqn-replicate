#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import code
from ale_python_interface import ALEInterface
from PIL import Image
from random import randrange
import scipy.misc

def buildDQN(action_num=4):
  inpt = tf.placeholder(tf.float32, shape=(None,84,84,4), name="input_layer")
  conv1 = tf.layers.conv2d(inputs=inpt, filters=16, kernel_size=[8,8], strides=4, activation=tf.nn.relu)
  conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4,4], strides=2, activation=tf.nn.relu)
  conv2_r = tf.reshape(conv2, [-1, 9*9*32])
  dense = tf.layers.dense(inputs=conv2_r, units=256, activation=tf.nn.relu)
  
  linear_W = tf.Variable(tf.truncated_normal([256, action_num]))
  outpt = tf.matmul(dense, linear_W)
  return outpt, inpt


def saveScreenShot(screenshot):
    for i in range(len(screenshot)):
      im = Image.fromarray(screenshot[:,:,i])
      im.save("file" + str(i) + ".jpg")

def performAction(ale, action):
  reward = []
  screenshot = []
  legal_actions = ale.getMinimalActionSet()
  for i in range(1,5):
    action =  legal_actions[randrange(len(legal_actions))]
    reward += [ale.act(action)] 
    #screenshot += [ale.getScreenGrayscale()]
    screenshot += [scipy.misc.imresize(ale.getScreenGrayscale()[:,:,0], (110,84))[18:102,:]]
  screenshot = np.array(screenshot).swapaxes(0,2).swapaxes(0,1)
  print(screenshot.shape)
  saveScreenShot(screenshot)
  print(reward)

ins = np.random.rand(10,84,84,4)
network, inpt = buildDQN()

ale = ALEInterface()
ale.setInt(b'random_seed', 123)
ale.loadROM(str.encode("/home/tim/space_invaders.bin"))

legal_actions = ale.getMinimalActionSet()
network, inpt = buildDQN(len(legal_actions))
performAction(ale, legal_actions[1])

#code.interact(local=locals())


#optimizer = tf.train.GradientDescentOptimizer(0.001)
#
#target = tf.constant([0.3,-0.4,0.2,-0.5])
#loss = tf.nn.l2_loss(target - network)
#tf.summary.scalar('loss',loss)
#train_step = optimizer.minimize(loss)
#
##tensorboard
#merged = tf.summary.merge_all()
#t_writer = tf.summary.FileWriter("/tmp/test")
#
#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#
#for i in range(10):
#  summ, _ = sess.run([merged, train_step], feed_dict={inpt: ins})
#  t_writer.add_summary(summ,i)
#  result = sess.run(network, feed_dict={inpt: ins})
#  print(result)
#code.interact(local=locals())
