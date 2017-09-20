import random
import numpy as np
import code
import pickle

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
      self.data[random.randrange(self.current_size)] = (phi_t, a_t, r_t, phi_t1)
  def random_minibatch(self, minibatch_size=32):
     minibatch = random.sample(self.data, minibatch_size)
     phi_t, a_t, r_t, phi_t1 = list(map(list, zip(*minibatch)))
     return (np.array(phi_t), np.array(a_t), np.array(r_t), np.array(phi_t1))
  def save_data(self, name="replayMemory.dat"):
    with open(name,'wb') as f:
      pickle.dump(self.data, f)
    import pdb; pdb.set_trace()
  def load_data(self, name="replayMemory.dat"):
    with open(name,'rb') as f:
      self.data = pickle.load(f)
