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
from a3c_model import A3CModel, loggerConfig, SynthA3CModel
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime
from synth import SynthGame
import distance

dtype = torch.FloatTensor

T_max = 100000
t_max = 5
T = Value('i',1)
gamma = 0.99
beta = 0.01
 

def play_game(num, shared_model, gradient_queue,  log_queue, parameters_lock, logger):
    legal_actions = list(range(7))
    game = SynthGame()
    t = 1
    T.value += 1 #Not synchornized, but doesn't really matter 

    model = SynthA3CModel(len(legal_actions))


    st = game.nnexpr.get_windowed_state()
    over = False
    total_reward = 0
    action_log = []
    while T.value < T_max:
        model.zero_grad()
        #synchronize network params
        with parameters_lock:
          model.load_state_dict(shared_model)
        logger.debug("%d Updated parameters %e", num,  next(model.parameters()).data.view(-1)[0])
        t_start = t
        rs = []
        while (not over) and t - t_start < t_max:
#           action = random_action(legal_actions) #TODO used pi for this
           st = Variable(torch.from_numpy(st).type(dtype))
           policy_probabilities, value = model(st)
           policy_probabilities = policy_probabilities.view(-1)
           action = policy_probabilities.multinomial().data[0]

           st, r, over = game.act(action)
           action_log += [(r, action)]
           total_reward += r
           rs += [(r, policy_probabilities, action, st, value) ]
           t, T.value = t + 1, T.value + 1
        R = 0 if over else value.data.numpy().flatten()[0]
        logger.debug("%d Used parameters %e", num,next(model.parameters()).data.view(-1)[0])
        for ri, pi_i, ai, si, V_si in rs:
            R = ri + gamma*R
            V_si = V_si.view(1)
            #import pdb; pdb.set_trace()
            entropy = -1 * torch.sum(torch.log(pi_i) * pi_i)
            pi_loss = torch.log(pi_i[ai]) * (float(R) - V_si) - beta*entropy
            V_loss = torch.pow(float(R) - V_si,2)
            total_loss = pi_loss + 0.5*V_loss
            total_loss.backward()
        log_queue.put(("loss/policy", T.value,pi_loss.data[0]))
        log_queue.put(("loss/value", T.value, V_loss.data[0]))
        log_queue.put(("loss/entropy", T.value, entropy.data[0]))
        log_queue.put(("value", T.value, V_si.data[0]))
        #Update global grad
        torch.nn.utils.clip_grad_norm(model.parameters(),40)
        grads = [params.grad for params in model.parameters()]
        gradient_queue.put(grads)

        if over:
            over = False
            rs = []
            logger.warning("%d     GAME OVER     %.4f",num, total_reward)
            if total_reward > 0.0:
              logger.warning("n: %s = %d; %d\ni: %s = %d; %d",
                 str(game.nnexpr), game.nnexpr.evalExpr(),len(str(game.nnexpr)), 
                 game.harness.initial_expr, game.harness.initial_value, len(game.harness.initial_expr))
              logger.warning(str(action_log))
            log_queue.put(("workers/totalReward", T.value,  total_reward))
            log_queue.put(("workers/gameTime", T.value, t))
            game.reset()
            action_log = []
            t = 0
            total_reward = 0
            st = game.nnexpr.get_windowed_state()


learning_rate = 1e-3
weight_decay = 0.99

def master_thread(gradient_queue, log_queue, logger):
  num_legal_actions = 7 #need to manually change this
  parameters_lock = Lock()
  shared_model = SynthA3CModel(num_legal_actions)
  shared_model.share_memory()
  shared_model_state = shared_model.state_dict()
  optimizer = torch.optim.RMSprop(shared_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  proc_num = 3
  procs = [Process(target=play_game, args=(i, shared_model_state, gradient_queue, log_queue, parameters_lock, logger)) for i in range(proc_num)]
  list(map(lambda p: p.start(), procs))
  logger.info("spawned processes")
  while T.value < T_max:
      logger.debug("Waiting for gradients, current parameter %e", next(shared_model.parameters()).data.view(-1)[0])
      grads = gradient_queue.get()
      if T.value % 25 == 0:
        logger.warning("Updating grads %e %s", grads[0].data.view(-1)[0], "========" if grads[0].data.view(-1)[0] != 0.0 else "")
      for gradients, (name, params) in zip(grads, shared_model.named_parameters()):
          params.grad = gradients
          #log_queue.put(('master/gradients/{}'.format(name), T.value, gradients.norm().data[0]))
      with parameters_lock:
        optimizer.step()
  import pdb; pdb.set_trace()

  
def logger_thread(log_queue):
  log_dir = '/tmp/runs/synth-{}'.format(str(datetime.now())[5:16])
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

code.interact(local=locals())
