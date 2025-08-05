import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class SoftmaxRegression(d2l.Classifier):
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))

    #loss function (cross entropy) defined in d2l