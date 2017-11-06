"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

def fit_recurrent(train_loader,val_loader,model,
                  optimizer='adam',loss_fcn=nn.NLLLoss,
                  learnrate=1e-3,cuda=True,max_epochs=200):

    if cuda:
        model = model.cuda()

    optimizer = {'adam':torch.optim.Adam(model.parameters(),lr=learnrate),
                 'sgd':torch.optim.SGD(model.parameters(),lr=learnrate,
                                       momentum=0.9),
                 'adamax':torch.optim.Adamax(model.parameters(),lr=learnrate)
                 }[optimizer.lower()]

    def train_epoch():
        def train_batch(x,y):
            hidden = model.init_hidden()
            optimizer.zero_grad()
            for i in range(x.size()[0]):
                output,hidden = model(x[i],hidden)
            loss = loss_fcn(output,y)
            loss.backward()
            optimizer.step()
            return loss.data

        for x,y in train_loader:
            if cuda:
                x,y = x.cuda(),y.cuda()
            x,y = Variable(x),Variable(y)
            loss = train_batch(x,y)
            print(loss)

    def val_epoch():
        def val_batch(x,y):
            hidden = model.init_hidden()
            for i in range(x.size()[0]):
                output,hidden = model(x[i],hidden)
            loss = loss_fcn(output,y)
            return loss.data

        for x,y in val_loader:
            if cuda:
                x,y = x.cuda(),y.cuda()
            x,y = Variable(x),Variable(y)
            loss = val_batch(x,y)
            print(loss)








