"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import numpy as np
import progressbar
import torch
from torch import nn
from torch.autograd import Variable


def fit_recurrent(train_loader,val_loader,model,
                  optimizer='adam',loss_fcn=nn.NLLLoss,
                  learnrate=1e-3,cuda=True,patience=20,max_epochs=200):

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
            correct = 0
            for i in range(x.size()[1]):
                output,hidden = model(x[i],hidden)
            loss = loss_fcn(output,y)
            loss.backward()
            optimizer.step()
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            return loss.data,correct

        bar = progressbar.ProgressBar()
        losses = []
        correct = 0
        for x,y in bar(train_loader):
            if cuda:
                x,y = x.cuda(),y.cuda()
            x,y = Variable(x),Variable(y)
            loss,corr = train_batch(x,y)
            losses.append(loss)
            correct += corr
        return np.mean(losses),float(correct)/len(train_loader.dataset)

    def val_epoch():
        def val_batch(x,y):
            hidden = model.init_hidden()
            correct = 0
            for i in range(x.size()[1]):
                output,hidden = model(x[i],hidden)
            loss = loss_fcn(output,y)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            return loss.data,correct

        bar = progressbar.ProgressBar()
        losses = []
        correct = 0
        for x,y in bar(val_loader):
            if cuda:
                x,y = x.cuda(),y.cuda()
            x,y = Variable(x),Variable(y)
            loss,corr = val_batch(x,y)
            losses.append(loss)
            correct += corr
        return np.mean(losses),float(correct)/len(train_loader.dataset)

    train_losses = []
    val_losses = []
    best_val = np.inf
    stall = 0
    for epoch in range(max_epochs):
        train_losses.append(np.mean(train_epoch()))
        val_losses.append(np.mean(val_epoch()))
        print('Epoch %i   -   Training loss = %8.4f   - Validation loss = %8.4f'
              %(epoch,train_losses[-1],val_losses[-1]))
        if val_losses[-1]<best_val:
            best_val = val_losses[-1]
            stall = 0
        else:
            stall += 1
        if stall>=patience:
            break






