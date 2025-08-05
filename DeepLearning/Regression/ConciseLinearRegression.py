import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import ModuleClass
import TrainerClass
import DataModuleClass
from DataModuleClass import Data

class LinearRegression(d2l.Module):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        return self.net(X)
    
    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
    
    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)


def l2_penalty(w):
    return (w ** 2).sum() / 2

class WeightDecay(LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
    
    def configure_optimizers(self):
        return torch.optim.SGD([{'params': self.net.weight, 'weight decay': self.wd}, {'params': self.net.bias}], lr = self.lr)

# model = LinearRegression(lr=0.03)
# data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
# trainer = d2l.Trainer(max_epochs=3)
# 

data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))