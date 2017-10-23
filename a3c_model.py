import torch
from ale_python_interface import ALEInterface
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging


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
  def __init__(self, num_actions):
    super(A3CModel, self).__init__()
    self.convLayers =  torch.nn.Sequential(
                        torch.nn.Conv2d(1, 16,8, stride=4),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(16, 32,4, stride=2),
                        torch.nn.ReLU()
                       )
    self.linearLayer = torch.nn.Linear(2592,256)
    self.policyLinearLayer = torch.nn.Linear(256, num_actions)
    self.valueLinearLayer = torch.nn.Linear(256,1)
    self.softmax = torch.nn.Softmax()
  def forward(self, x):
    out_conv = self.convLayers(x)
    out_conv = out_conv.view(out_conv.size(0), -1)
    out = self.linearLayer(out_conv).clamp(min=0)
    policy = self.policyLinearLayer(out)
    policy = F.softmax(policy.view(-1))
    value = self.valueLinearLayer(out)
    return policy, value

def loggerConfig():
    logger = logging.getLogger()
    #formatter = logging.Formatter( '%(asctime)s %(levelname)-2s %(message)s')
    formatter = logging.Formatter( '%(message)s')
    streamhandler_ = logging.StreamHandler()
    streamhandler_.setFormatter(formatter)
    logger.addHandler(streamhandler_)
#    fileHandler_ = logging.FileHandler("log/a3c_training_log_"+ts)
#    fileHandler_.setFormatter(formatter)
#    logger.addHandler(fileHandler_)
    logger.setLevel(logging.INFO)
    return logger
#example = Variable(torch.from_numpy(get_example()).type(dtype))
