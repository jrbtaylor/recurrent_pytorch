"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import torch
import torch.nn as nn


class Norm(nn.Module):
    """
    Re-usable class for either batch-norm or layer-norm (by swapping dim)
    """
    def __init__(self,n_hidden,eps=1e-8,dim=0):
        super(Norm,self).__init__()
        self.eps = eps
        self.n_hidden = n_hidden
        self.a = nn.Parameter(torch.ones(1,n_hidden),requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1,n_hidden),requires_grad=True)
        self.dim = dim

    def forward(self,x):
        mean_x = torch.mean(x,dim=self.dim).expand_as(x)
        std_x = torch.std(x,dim=self.dim).expand_as(x)
        out = (x-mean_x)/(std_x+self.eps)
        out = out*self.a.expand_as(x)+self.b.expand_as(x)
        return out


class LayerNorm(Norm):
    def __init__(self,n_hidden,eps=1e-8):
        super(LayerNorm,self).__init__(n_hidden,eps,dim=1)


class BatchNorm(Norm):
    def __init__(self,n_hidden,eps=1e-8):
        super(BatchNorm,self).__init__(n_hidden,eps,dim=0)


class RNN(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,layer_norm=False):
        super(RNN,self).__init__()
        self.n_hidden = n_hidden
        self.i2h = nn.Linear(n_in+n_hidden,n_hidden)
        self.dropout = nn.Dropout()
        self.h2o = nn.Linear(n_hidden,n_out)
        self.softmax = nn.LogSoftmax()
        self.layer_norm = layer_norm
        if layer_norm:
            self.normh = LayerNorm(n_hidden)
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()

    def forward(self,input,hidden):
        combined = torch.cat((input,hidden),1)
        hidden = self.i2h(combined)
        hidden = self.activation(hidden)
        if self.layer_norm:
            hidden = self.normh(hidden)
        output = self.h2o(self.dropout(hidden))
        output = self.softmax(output)
        return output,hidden

    def init_hidden(self,batch_size=1):
        return nn.Parameter(torch.zeros(1,self.n_hidden),
                            requires_grad=True).repeat(batch_size,1)


class GRU(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,layer_norm=False):
        super(GRU,self).__init__()
        self.n_hidden = n_hidden
        self.gate = nn.Sigmoid()
        self.i2g = nn.Linear(n_in+n_hidden,2*n_hidden,bias=True)
        self.activation = nn.Tanh()
        self.x2h = nn.Linear(n_in,n_hidden,bias=True)
        self.h2h = nn.Linear(n_hidden,n_hidden,bias=False)
        self.dropout = nn.Dropout()
        self.h2o = nn.Linear(n_hidden,n_out,bias=True)
        self.softmax = nn.LogSoftmax()

    def forward(self,input,hidden):
        combined = torch.cat((input,hidden),1)
        gates = self.gate(self.i2g(combined))
        z,r = torch.split(gates,split_size=self.n_hidden,dim=1)
        h = z*hidden+(1-z)*self.activation(self.x2h(input)+self.h2h(r*hidden))
        output = self.softmax(self.h2o(self.dropout(h)))
        return output,h

    def init_hidden(self,batch_size=1):
        return nn.Parameter(torch.zeros(1,self.n_hidden),
                            requires_grad=True).repeat(batch_size,1)
