#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import code
from ale_python_interface import ALEInterface
from PIL import Image
from random import randrange
import scipy.misc
import random

replay_memory_capacity = 1000
minibatch_size = 32

def buildDQN(action_num=4, reuse=False):
  with tf.variable_scope('Deep-Q-Net', reuse=reuse):
    inpt = tf.placeholder(tf.float32, shape=(None,84,84,4), name="input_layer")
    conv1 = tf.layers.conv2d(inputs=inpt, filters=16, kernel_size=[8,8], strides=4, reuse=reuse, activation=tf.nn.relu, name="Conv1_8x8")
    conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4,4], strides=2, activation=tf.nn.relu, name="Conv2_4x4")
    conv2_r = tf.reshape(conv2, [-1, 9*9*32])
    dense = tf.layers.dense(inputs=conv2_r, units=256, activation=tf.nn.relu, name="dense")
    
    linear_W = tf.get_variable("linear_layer", [256, action_num], tf.float32, tf.random_normal_initializer)
    outpt = tf.matmul(dense, linear_W)
    if not reuse:
      #tf.summary.image("minibatch", inpt)
      pass
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
    reward += [ale.act(action)] 
    #screenshot += [ale.getScreenGrayscale()]
    screenshot += [scipy.misc.imresize(ale.getScreenGrayscale()[:,:,0], (110,84))[18:102,:]]
  screenshot = np.array(screenshot).swapaxes(0,2).swapaxes(0,1)
#  print(screenshot.shape)
#  saveScreenShot(screenshot)
#  print(reward)
  return screenshot, np.sign(sum(reward))

def random_action(legal_actions):
  return  legal_actions[randrange(len(legal_actions))]

def init_replay_memory(ale):
  legal_actions = ale.getMinimalActionSet()
  D = []
  prev_state, r = performAction(ale, 0)
  for i in range(replay_memory_capacity):
    if ale.game_over():
      ale.reset_game()
    action = random_action(legal_actions)
    state, r = performAction(ale, action) 
    D += [(prev_state, action, r, state, ale.game_over())]
  
  ale.reset_game()
  return np.array(D)
      



ale = ALEInterface()
ale.setInt(b'random_seed', 123)
ale.loadROM(str.encode("/home/tim/space_invaders.bin"))

legal_actions = ale.getMinimalActionSet()

D = init_replay_memory(ale)

network, inpt = buildDQN(len(legal_actions))
Qj, Qj_inpt = network, inpt
Qj1, Qj1_inpt = buildDQN(len(legal_actions), reuse=True)

rj = tf.placeholder(tf.float32, shape=(None), name="rj")
is_terminal = tf.placeholder(tf.float32, shape=(None), name="is_terminal")
gamma = tf.constant(0.1)

yj = is_terminal*rj + (1 - is_terminal)*(rj + gamma*tf.reduce_max(Qj1))
loss = tf.nn.l2_loss(tf.tile(tf.reshape(yj, [-1,1]),[1, 6]) - Qj)
tf.summary.scalar('loss',loss)

optimizer = tf.train.RMSPropOptimizer(0.001)
train_step = optimizer.minimize(loss)

minibatch = D[np.random.choice(replay_memory_capacity,minibatch_size, replace=False)]
feed_dict={Qj_inpt: np.array( list(minibatch[:,0])), 
						    Qj1_inpt: np.array( list(minibatch[:,3])), 
						    rj: minibatch[:,2].astype(np.float32) , 
                                                    is_terminal: minibatch[:,4].astype(np.float32)}

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
##tensorboard
merged = tf.summary.merge_all()
t_writer = tf.summary.FileWriter("/tmp/test")
t_writer.add_graph(sess.graph)

#code.interact(local=locals())
#re = sess.run(network, feed_dict={inpt:list(minibatch[:,0])})
#re1 = sess.run(Qj1, feed_dict={Q:list(minibatch[:,0])})
#print(re)



M = 10
epsilon = 0.4
i = 0
for episode in range(M):
  st, _ = performAction(ale, 0)
  los = sess.run(loss, feed_dict={Qj_inpt: np.array( list(minibatch[:,0])), 
   					    Qj1_inpt: np.array( list(minibatch[:,3])), 
   					    rj: minibatch[:,2].astype(np.float32) , 
                                                is_terminal: minibatch[:,4].astype(np.float32)})
  print(str(episode) + "  loss: "  + str(los))
  T = 0
  while not ale.game_over():
    if random.random() < epsilon:
      action = random_action(legal_actions)
    else:
      result = sess.run(network, feed_dict={inpt: [st]})
#      import pdb; pdb.set_trace()
      action = legal_actions[np.argmax(result)]
    st1, r = performAction(ale, action) 
    D[randrange(replay_memory_capacity)] = (st, action, r, st1, ale.game_over()) 
    minibatch = D[np.random.choice(replay_memory_capacity,minibatch_size, replace=False)]
    st = st1


#
    summ, _ = sess.run([merged, train_step], feed_dict={Qj_inpt: np.array( list(minibatch[:,0])), 
						    Qj1_inpt: np.array( list(minibatch[:,3])), 
						    rj: minibatch[:,2].astype(np.float32) , 
                                                    is_terminal: minibatch[:,4].astype(np.float32)})

    #print("loss: " + str(los))
    i += 1
    t_writer.add_summary(summ, i)
    T += 1


  ale.reset_game() 


code.interact(local=locals())
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
