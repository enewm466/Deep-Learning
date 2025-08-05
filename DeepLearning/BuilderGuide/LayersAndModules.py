import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
X = torch.rand(2, 20)
#net = MLP()
#print(net(X).shape)

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    
    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X
    
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
print(net(X).shape)