#!/usr/bin/python3
import numpy as np
import code
from ale_python_interface import ALEInterface
from PIL import Image
import scipy.misc
from random import randrange
import random
from datetime import datetime
from multiprocessing import Process, Pool

replay_memory_capacity = 180000
minibatch_size = 32

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
  screenshot = np.array(screenshot).swapaxes(0,2).swapaxes(0,1)
#  print(screenshot.shape)
#  saveScreenShot(screenshot)
#  print(reward)
  return screenshot, np.sign(sum(reward)) + np.sign(ale.lives() - lives_before), sum(reward)

def random_action(legal_actions):
  return  legal_actions[randrange(len(legal_actions))]


def play_game(num):
    ale = ALEInterface()
    ale.setInt(b'random_seed', 123)
    ale.setBool(b'display_screen', True)
    ale.loadROM(str.encode("/homes/tk1713/space_invaders.bin"))
    legal_actions = ale.getMinimalActionSet()
    for E in range(100):
      while not ale.game_over():
         action = random_action(legal_actions)
         ale.act(action)

proc_num = 10
with Pool(proc_num) as p:
  p.map(play_game, range(proc_num))

#code.interact(local=locals())
