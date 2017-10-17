#!/usr/bin/python3
import numpy as np
import code
from ale_python_interface import ALEInterface
from PIL import Image
import scipy.misc
from random import randrange
import random
from datetime import datetime
from torch.multiprocessing import Process, Pool, Value, Queue
from a3c_model import get_model
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime

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
gamma = 0.99
 

def play_game(num, shared_models, gradient_queue,  log_queue):
    ale = ALEInterface()
    ale.setInt(b'random_seed', 123)
    ale.setBool(b'display_screen', True)
    ale.loadROM(str.encode("/homes/tk1713/space_invaders.bin"))
    legal_actions = ale.getMinimalActionSet()
    t = 1
    T.value += 1 #Not synchornized, but doesn't really matter 

    pi, V = get_model(len(legal_actions))
    pi_shared, V_shared = shared_models

    [ale.act(0) for i in range(131)] #skip start of the game
    st, r, r_full = performAction(ale, 0)
    total_game_reward = 0
    while T.value < T_max:
        pi.zero_grad()
        V.zero_grad()
        #synchronize network params
        pi.load_state_dict(pi_shared)
        V.load_state_dict(V_shared)
        print(num, "Updated parameters", next(pi.parameters()).data[0,0,0,0])
        t_start = t
        rs = []
        while (not ale.game_over()) and t - t_start < t_max:
#           action = random_action(legal_actions) #TODO used pi for this
           policy_probabilities = pi(st)
           np_pi = policy_probabilities.data.numpy()[0]
           action = np.random.choice(len(legal_actions), p=np_pi)
           st, r, r_full = performAction(ale, action)
           rs += [(r, policy_probabilities, action, st) ]
           t, T.value = t + 1, T.value + 1
           total_game_reward += r_full
        R = 0 if ale.game_over() else V(st).data.numpy().flatten()[0]
        print(num, "Used parameters", next(pi.parameters()).data[0,0,0,0])
        for ri, pi_i, ai, si in rs:
            R = ri + gamma*R
            V_si = V(si).view(1)
            #import pdb; pdb.set_trace()
            pi_loss = torch.log(pi_i[0,ai] * (float(R) - V_si))
            V_loss = torch.pow(float(R) - V_si,2)
            pi_loss.backward(retain_graph=True)
            V_loss.backward()
            #Acc grad
        #Update global grad
        pi_grads = [pi_params.grad for pi_params in pi.parameters()]
        V_grads = [V_params.grad for name, V_params in V.named_parameters() if name.startswith('1.')]
#        print(num, "Sending gradients")
        gradient_queue.put((pi_grads, V_grads))

        if ale.game_over():
            ale.reset_game()
            rs = []
            print(num, "     GAME OVER     ", total_game_reward)
            log_queue.put(("workers/totalReward", T.value, total_game_reward))
            log_queue.put(("worker/{}/totalReward".format(num), T.value, total_game_reward))
            total_game_reward = 0
            [ale.act(0) for i in range(131)] #skip start of the game
            st, r, r_full = performAction(ale, 0)


    for E in range(100):
      while not ale.game_over():
         action = random_action(legal_actions)
         ale.act(action)

learning_rate = 1e-6

def master_thread(gradient_queue, log_queue):
  num_legal_actions = 6 #need to manually change this
  pi, V = get_model(num_legal_actions)
  pi.share_memory()
  V.share_memory()
  shared_model_state = (pi.state_dict(), V.state_dict())
  pi_optimizer = torch.optim.RMSprop(pi.parameters(), lr=learning_rate)
  V_optimizer = torch.optim.RMSprop([p for name, p in V.named_parameters() if name.startswith('1.')], lr=learning_rate)
  proc_num = 4
#  with Pool(proc_num) as p:
#_    p.map_async(play_game, [(i, shared_model_state, gradient_queue) for i in range(proc_num)])
  procs = [Process(target=play_game, args=(i, shared_model_state, gradient_queue, log_queue)) for i in range(proc_num)]
  list(map(lambda p: p.start(), procs))
  print("spawned processes")
  while T.value < T_max:
      print("Waiting for gradients, current parameter", next(pi.parameters()).data[0,0,0,0])
      pi_grads, V_grads = gradient_queue.get()
      print("Updating pi grads", pi_grads[0].data[0,0,0,0], "========" if pi_grads[0].data[0,0,0,0] != 0.0 else "")
      for pi_g, (name, pi_param) in zip(pi_grads, pi.named_parameters()):
          #pi_param.data += learning_rate * pi_g.data
          pi_param.grad = pi_g
          log_queue.put(('master/gradients/{}'.format(name), T.value, pi_g.norm().data[0]))
      pi_optimizer.step()
      for V_g, V_param in zip(V_grads, [p for name, p in V.named_parameters() if name.startswith('1.')]):
          V_param.grad =  V_g
      V_optimizer.step()

  
def logger_thread(log_queue):
  log_dir = '/tmp/runs/{}'.format(str(datetime.now())[5:16])
  print(log_dir)
  writer = SummaryWriter(log_dir=log_dir)
  while T.value < T_max:
      name, step, value = log_queue.get()
      writer.add_scalar(name, value, global_step=step)


gradient_queue = Queue()
log_queue = Queue()

if __name__ == '__main__':
  log_proc = Process(target=logger_thread, args=(log_queue,))
  log_proc.start()
  master_thread(gradient_queue, log_queue)
  #play_game(0)

#code.interact(local=locals())
