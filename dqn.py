#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import code
from ale_python_interface import ALEInterface
from PIL import Image
import scipy.misc
from random import randrange
import random
from datetime import datetime
from ReplayMemory import ReplayMemory

replay_memory_capacity = 80000
minibatch_size = 32

def buildDQN(action_num=4, reuse=False):
  with tf.variable_scope('Deep-Q-Net', reuse=reuse):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    inpt = tf.placeholder(tf.float32, shape=(None,84,84,4), name="input_layer")
    conv1 = tf.layers.conv2d(inputs=inpt, filters=16, kernel_size=[8,8], strides=4, reuse=reuse, activation=tf.nn.relu, name="Conv1_8x8")
    conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4,4], strides=2, activation=tf.nn.relu, reuse=reuse,name="Conv2_4x4")
    conv2_r = tf.reshape(conv2, [-1, 9*9*32])
    dense = tf.layers.dense(inputs=conv2_r, units=256, activation=tf.nn.relu, kernel_regularizer=regularizer, name="dense", reuse=reuse)
    
    linear_W = tf.get_variable("linear_layer_weights", [256, action_num], tf.float32, tf.random_normal_initializer, regularizer=regularizer)
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
  return screenshot, np.sign(sum(reward)), sum(reward)

def random_action(legal_actions):
  return  legal_actions[randrange(len(legal_actions))]

def init_replay_memory(ale, D):
  legal_actions = ale.getMinimalActionSet()
  prev_state, r, _ = performAction(ale, 0)
  for i in range(10000):
    if ale.game_over():
      ale.reset_game()
    action = random_action(legal_actions)
    state, r, _ = performAction(ale, action) 
    D.add(prev_state, action, r, state)
  
  ale.reset_game()
      
def action_to_index(actions, legal_actions):
  return np.array(list(map(lambda a: [1, np.where(a == legal_actions)[0][0]], actions)))


ale = ALEInterface()
ale.setInt(b'random_seed', 123)
ale.loadROM(str.encode("/home/tim/space_invaders.bin"))
#ale.loadROM(str.encode("/home/tim/breakout.bin"))
#ale.loadROM(str.encode("/home/tim/Seaquest.A26"))
legal_actions = ale.getMinimalActionSet()

D = ReplayMemory(replay_memory_capacity)
init_replay_memory(ale, D)

Qj, Qj_inpt = buildDQN(len(legal_actions))

yj = tf.placeholder(tf.float32, shape=(None), name="yj")
acti = tf.placeholder(tf.int32, shape=(None), name="action")

with tf.name_scope("loss"):
  loss = tf.reduce_mean(tf.square(tf.subtract(yj,tf.gather_nd(Qj,acti))))
  #loss = tf.reduce_mean(tf.square(tf.clip_by_value(tf.subtract(yj,tf.gather_nd(Qj,acti)), -1, 1)))
  tf.summary.scalar('loss',loss)

optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99,0.0, 1e-6)
train_step = optimizer.minimize(loss)


control_states = D.random_minibatch(200)[0]

sess = tf.Session()
sess_hat = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

##tensorboard
merged = tf.summary.merge_all()
t_writer = tf.summary.FileWriter("/tmp/test1/" + str(datetime.now())[5:16])
t_writer.add_graph(sess.graph)
saver = tf.train.Saver()


#code.interact(local=locals())
#import pdb; pdb.set_trace()


M = 600
epsilon = 0.9
gamma = 0.99
i, total_reward, T, q_val_metric = 0, 0, 0, 0
for episode in range(M):
  st, _ , _ = performAction(ale, 0)
  print(str(episode) + "  epsilon: "  + str(epsilon))
  while not ale.game_over():
    if i % 1000 == 0:
      print("Switching over variables")
      saver.save(sess, "hat_vars.ckpt")
      saver.restore(sess_hat, "hat_vars.ckpt")
    if random.random() < epsilon:
      action = random_action(legal_actions)
    else:
      result = sess.run(Qj, feed_dict={Qj_inpt: [st]})
      action = legal_actions[np.argmax(result)]
    st1, r, rew = performAction(ale, action) 
    total_reward += rew
    D.add(st, action, r, st1) 
    Qjs, ajs, rjs, Qj1s  = D.random_minibatch(minibatch_size)
    yjs = rjs
    if not ale.game_over():
      yjs = yjs + gamma*np.amax(sess_hat.run(Qj, feed_dict={Qj_inpt: Qj1s}), axis=1)
    st = st1
    summ, _ = sess.run([merged, train_step], feed_dict={Qj_inpt: Qjs, yj: yjs, acti: action_to_index(ajs, legal_actions)})

    if T % 50 == 0 and i > 500:
      q_vals = sess.run(Qj, feed_dict={Qj_inpt: control_states})
      q_val_metric = q_vals.max(axis=1).mean()
      q_val_summary = tf.Summary(value=[tf.Summary.Value(tag="Q Value metric", simple_value=q_val_metric)])
      t_writer.add_summary(q_val_summary, i)
#      print(q_vals[8])

    i += 1
    t_writer.add_summary(summ, i)
    T += 1

  print("T is: " + str(T) + " Q-val is: " + str(q_val_metric))
  total_rew_summary = tf.Summary(value=[tf.Summary.Value(tag="total_reward", simple_value=total_reward)])
  t_writer.add_summary(total_rew_summary, i)
  print("total reward: " + str(total_reward))
  T, total_reward = 0, 0
  epsilon = epsilon - 0.008 if epsilon > 0.11 else 0.1
  ale.reset_game() 


code.interact(local=locals())
performAction(ale, legal_actions[1])
#code.interact(local=locals())
