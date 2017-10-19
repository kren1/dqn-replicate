#!/usr/bin/python3
import numpy as np
import code
from ale_python_interface import ALEInterface
from PIL import Image
import scipy.misc
from random import randrange
import random
from datetime import datetime
from torch.multiprocessing import Process, Pool, Value, Queue, SimpleQueue, Lock
from a3c_model import A3CModel, loggerConfig
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

T_max = 10000000
t_max = 5
T = Value('i',1)
gamma = 0.99
beta = 0.99
 

def play_game(num, shared_model, gradient_queue,  log_queue, parameters_lock, logger):
    ale = ALEInterface()
    ale.setInt(b'random_seed', 23*num + 153)
#    ale.setBool(b'display_screen', True)
    ale.loadROM(str.encode("/homes/tk1713/space_invaders.bin"))
    legal_actions = ale.getMinimalActionSet()
    t = 1
    T.value += 1 #Not synchornized, but doesn't really matter 

    model = A3CModel(len(legal_actions))

    [ale.act(0) for i in range(131)] #skip start of the game
    st, r, r_full = performAction(ale, 0)
    total_game_reward = 0
    while T.value < T_max:
        model.zero_grad()
        #synchronize network params
        with parameters_lock:
          model.load_state_dict(shared_model)
        logger.debug("%d Updated parameters %e", num,  next(model.parameters()).data[0,0,0,0])
        t_start = t
        rs = []
        while (not ale.game_over()) and t - t_start < t_max:
#           action = random_action(legal_actions) #TODO used pi for this
           policy_probabilities, value = model(st)
           np_pi = policy_probabilities.data.numpy()[0]
           action = np.random.choice(len(legal_actions), p=np_pi)
           st, r, r_full = performAction(ale, action)
           rs += [(r, policy_probabilities, action, st, value) ]
           t, T.value = t + 1, T.value + 1
           total_game_reward += r_full
        R = 0 if ale.game_over() else value.data.numpy().flatten()[0]
        logger.debug("%d Used parameters %e", num,next(model.parameters()).data[0,0,0,0])
        for ri, pi_i, ai, si, V_si in rs:
            R = ri + gamma*R
            V_si = V_si.view(1)
            #import pdb; pdb.set_trace()
            entropy = torch.sum(torch.log(pi_i) * pi_i)
            pi_loss = torch.log(pi_i[0,ai]) * (float(R) - V_si) - beta*entropy
            V_loss = torch.pow(float(R) - V_si,2)
            total_loss = pi_loss + V_loss
            total_loss.backward()
        #Update global grad
        torch.nn.utils.clip_grad_norm(model.parameters(),40)
        grads = [params.grad for params in model.parameters()]
        gradient_queue.put(grads)

        if ale.game_over():
            ale.reset_game()
            rs = []
            logger.warning("%d     GAME OVER     %d",num, total_game_reward)
            log_queue.put(("workers/totalReward", T.value, total_game_reward))
            log_queue.put(("worker/{}/totalReward".format(num), T.value, total_game_reward))
            total_game_reward = 0
            [ale.act(0) for i in range(131)] #skip start of the game
            st, r, r_full = performAction(ale, 0)


    for E in range(100):
      while not ale.game_over():
         action = random_action(legal_actions)
         ale.act(action)

learning_rate = 1e-3
weight_decay = 0.99

def master_thread(gradient_queue, log_queue, logger):
  num_legal_actions = 6 #need to manually change this
  parameters_lock = Lock()
  shared_model = A3CModel(num_legal_actions)
  shared_model.share_memory()
  shared_model_state = shared_model.state_dict()
  optimizer = torch.optim.RMSprop(shared_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  proc_num = 8
  procs = [Process(target=play_game, args=(i, shared_model_state, gradient_queue, log_queue, parameters_lock, logger)) for i in range(proc_num)]
  list(map(lambda p: p.start(), procs))
  logger.info("spawned processes")
  while T.value < T_max:
      logger.info("Waiting for gradients, current parameter %e", next(shared_model.parameters()).data[0,0,0,0])
      grads = gradient_queue.get()
      logger.info("Updating grads %e %s", grads[0].data[0,0,0,0], "========" if grads[0].data[0,0,0,0] != 0.0 else "")
      for gradients, (name, params) in zip(grads, shared_model.named_parameters()):
          params.grad = gradients
          #log_queue.put(('master/gradients/{}'.format(name), T.value, gradients.norm().data[0]))
      with parameters_lock:
        optimizer.step()

  
def logger_thread(log_queue):
  log_dir = '/tmp/runs/{}'.format(str(datetime.now())[5:16])
  print(log_dir)
  writer = SummaryWriter(log_dir=log_dir)
  while T.value < T_max:
      name, step, value = log_queue.get()
      writer.add_scalar(name, value, global_step=step)


gradient_queue = SimpleQueue()
log_queue = SimpleQueue()

if __name__ == '__main__':
  log_proc = Process(target=logger_thread, args=(log_queue,))
  log_proc.start()
  master_thread(gradient_queue, log_queue, loggerConfig())
  #play_game(0)

#code.interact(local=locals())
