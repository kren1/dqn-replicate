#!/usr/bin/python3
import numpy as np
import code
from ale_python_interface import ALEInterface
from PIL import Image
import scipy.misc
from random import randrange
import random
from datetime import datetime
from multiprocessing import Process, Pool, Value
from a3c_model import get_model
import torch
from torch.autograd import Variable
dtype = torch.FloatTensor

def saveScreenShot(screenshot):
    for i in range(len(screenshot)):
      im = Image.fromarray(screenshot[:,:,i])
      im.save("file" + str(i) + ".jpg")

def performAction(ale, action):
  reward = []
  screenshot = []
  legal_actions = ale.getMinimalActionSet()
  lives_before = ale.lives()
  for i in range(1,5):
    reward += [ale.act(action)] 
    reward += [ale.act(action)] 
    reward += [ale.act(action)] 
    #screenshot += [ale.getScreenGrayscale()]
    screenshot += [scipy.misc.imresize(ale.getScreenGrayscale()[:,:,0], (110,84))[18:102,:]]
  screenshot = Variable(torch.from_numpy(np.array([np.array(screenshot).swapaxes(0,2).swapaxes(0,1).swapaxes(0,2)])).type(dtype))
#  print(screenshot.shape)
#  saveScreenShot(screenshot)
#  print(reward)
  if lives_before > ale.lives():
    return screenshot, -10, sum(reward)

  return screenshot, np.sign(sum(reward)), sum(reward)

def random_action(legal_actions):
  return  legal_actions[randrange(len(legal_actions))]

T_max = 10000
t_max = 10
T = Value('i',1)
gamma = 0.001


def play_game(num):
    ale = ALEInterface()
    ale.setInt(b'random_seed', 123)
    ale.setBool(b'display_screen', True)
    ale.loadROM(str.encode("/homes/tk1713/space_invaders.bin"))
    legal_actions = ale.getMinimalActionSet()
    t = 1
    T.value += 1 #Not synchornized, but doesn't really matter 
    pi, V = get_model(len(legal_actions))
    while T.value < T_max:
        #TODO: reset gradients
        #TODO: synchronize network params
        t_start = t
        [ale.act(0) for i in range(131)] #skip start of the game
        st, r, r_full = performAction(ale, 0)
        rs = []
        while (not ale.game_over()) and t - t_start < t_max:
#           action = random_action(legal_actions) #TODO used pi for this
           policy_probabilities = pi(st)
           action = np.random.choice(len(legal_actions), p=policy_probabilities.data.numpy()[0])
           st, r, r_full = performAction(ale, action)
           rs += [r]
           t, T.value = t + 1, T.value + 1
        R = 0 if ale.game_over() else V(st).data.numpy().flatten()[0]
        print(R, rs)
        for ri in rs:
            R = ri + gamma*R
            #Acc grad
        #Update global grad
        print(R, rs)
        if ale.game_over():
            ale.reset_game()
            rs = []


    for E in range(100):
      while not ale.game_over():
         action = random_action(legal_actions)
         ale.act(action)


if __name__ == '__main__':
  play_game(0)
  proc_num = 1
  with Pool(proc_num) as p:
    #p.map(play_game, range(proc_num))
    pass

#code.interact(local=locals())
