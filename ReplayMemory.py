import random
import numpy as np

class ReplayMemory:
  def __init__(self, max_size=1000):
    self.data = []
    self.current_size = 0
    self.max_size = max_size
  def add(self, phi_t, a_t, r_t, phi_t1):
    if self.current_size < self.max_size:
      self.data.append( (phi_t, a_t, r_t, phi_t1) )
      self.current_size += 1
    else:
      self.data[randrange(self.current_size)] = (phi_t, a_t, r_t, phi_t1)
  def random_minibatch(self, minibatch_size=32):
     minibatch = random.sample(self.data, minibatch_size)
     return np.swapaxes(np.array(minibatch) ,0,1)
     
