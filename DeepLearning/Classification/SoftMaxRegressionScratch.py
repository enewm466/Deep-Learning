import torch
from d2l import torch as d2l


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims = True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()

class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size = (num_inputs, num_outputs), requires_grad = True)
        self.b = torch.zeros(num_outputs, requires_grad = True)
    
    def parameters(self):
        return [self.W, self.b]
                
    def forward(self, X):
        X = X.reshape((-1, self.W.shape[0]))
        return softmax(torch.matmul(X, self.W) + self.b)
    
    def loss(self, y_hat, y):
        return cross_entropy(y_hat, y)

    