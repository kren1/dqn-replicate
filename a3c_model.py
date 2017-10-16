import torch
#from a3c import performAction
from ale_python_interface import ALEInterface
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

dtype = torch.FloatTensor

def get_example():
    ale = ALEInterface()
    ale.setInt(b'random_seed', 123)
    ale.setBool(b'display_screen', True)
    ale.loadROM(str.encode("/homes/tk1713/space_invaders.bin"))
    legal_actions = ale.getMinimalActionSet()
    st, r, total_r = performAction(ale, 0)
    return np.array([np.swapaxes(st,0,2)])


class A3CModel(torch.nn.Module):
  def __init__(self):
    super(A3CModel, self).__init__()
    self.convLayers =  torch.nn.Sequential(
                        torch.nn.Conv2d(4, 16,8, stride=4),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(16, 32,4, stride=2),
                        torch.nn.ReLU()
                       )
    self.linearLayer = torch.nn.Linear(2592,256)
  def forward(self, x):
    out_conv = self.convLayers(x)
    out_conv = out_conv.view(out_conv.size(0), -1)
    out = self.linearLayer(out_conv).clamp(min=0)
    return out

class PolicyModel(torch.nn.Module):
  def __init__(self, common_model, num_actions):
    super(PolicyModel, self).__init__()
    self.common_model = common_model
    self.linearLayer = torch.nn.Linear(256, num_actions)
    self.softmax = torch.nn.Softmax()
  def forward(self, x):
    out = self.common_model(x)
    out = self.linearLayer(out)
    out = self.softmax(out)
    return out


def get_model(num_actions):
  common_model = A3CModel()
  policy = PolicyModel(common_model, num_actions)
  value = torch.nn.Sequential(common_model, torch.nn.Linear(256,1))
  return policy, value



#example = Variable(torch.from_numpy(get_example()).type(dtype))
