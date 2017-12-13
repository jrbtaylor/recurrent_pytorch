"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import json
import os
import time

import numpy as np
import progressbar
import torch
from torch import nn
from torch.autograd import Variable

from plot import plot_stats


def _clearline():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)


def fit_recurrent(train_loader,val_loader,model,exp_path,
                  zoneout=0,optimizer='adam',loss_fcn=nn.NLLLoss(),
                  aux_weight=0.1,learnrate=1e-3,cuda=True,
                  patience=20,max_epochs=200,resume=False):
    if cuda:
        model = model.cuda()

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    statsfile = os.path.join(exp_path,'stats.json')

    optimizer = {'adam':torch.optim.Adam(model.parameters(),lr=learnrate),
                 'sgd':torch.optim.SGD(
                     model.parameters(),lr=learnrate,momentum=0.9),
                 'adamax':torch.optim.Adamax(model.parameters(),lr=learnrate)
                 }[optimizer.lower()]

    # TO-DO: extend this to track other things (gradient magnitudes, etc.)
    if not resume:
        stats = {'loss':{'train':[],'val':[]},
                 'acc':{'train':[],'val':[]},
                 'aux_loss':{'train':[],'val':[]}}
        best_val = np.inf
        stall = 0
        start_epoch = 0
    else:
        with open(statsfile,'r') as js:
            stats = json.load(js)
        best_val = np.min(stats['loss']['val'])
        stall = len(stats['loss']['val'])-np.argmin(stats['loss']['val'])-1
        start_epoch = len(stats['loss']['val'])

    def train_epoch():
        def train_batch(x,y):
            model.dropout.p = 0.5
            hidden = model.init_hidden(x.data.size()[0])
            p_zoneout = zoneout*torch.ones(hidden.data.size())
            if cuda:
                hidden = hidden.cuda()
                p_zoneout = p_zoneout.cuda()
            optimizer.zero_grad()
            correct = 0
            aux_losses = []
            for i in range(x.size()[1]):
                output,h_new = model(x[:,i],hidden)
                aux_losses.append(model.aux_loss)
                zoneout_mask = Variable(torch.bernoulli(p_zoneout))
                hidden = zoneout_mask*hidden+(1-zoneout_mask)*h_new
            main_loss = loss_fcn(output,y)
            loss = main_loss+aux_weight*np.mean(aux_losses)
            loss.backward()
            optimizer.step()
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            return main_loss.data.cpu().numpy(),correct,np.mean(aux_losses)

        bar = progressbar.ProgressBar()
        losses = []
        correct = 0
        aux_losses = []
        for x,y in bar(train_loader):
            if cuda:
                x,y = x.cuda(),y.cuda()
            x,y = Variable(x),Variable(y)
            loss,corr,aux = train_batch(x,y)
            losses.append(loss)
            correct += corr
            aux_losses.append(aux)
        _clearline()
        loss = float(np.mean(losses))
        accuracy = float(correct)/len(train_loader.dataset)
        aux_loss = float(np.mean(aux_losses))
        return loss, accuracy, aux_loss

    def val_epoch():
        def val_batch(x,y):
            model.dropout.p = 0
            hidden = model.init_hidden(x.data.size()[0])
            p_zoneout = zoneout*torch.ones(hidden.data.size())
            if cuda:
                hidden = hidden.cuda()
                p_zoneout = p_zoneout.cuda()
            correct = 0
            aux_losses = []
            for i in range(x.size()[1]):
                output,h_new = model(x[:,i],hidden)
                aux_losses.append(model.aux_loss)
                # take expectation of zoneout
                hidden = zoneout*hidden+(1-zoneout)*h_new
            main_loss = loss_fcn(output, y)
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            return main_loss.data.cpu().numpy(),correct,np.mean(aux_losses)

        bar = progressbar.ProgressBar()
        losses = []
        correct = 0
        aux_losses = []
        for x,y in bar(val_loader):
            if cuda:
                x,y = x.cuda(),y.cuda()
            x,y = Variable(x),Variable(y)
            loss,corr,aux = val_batch(x,y)
            losses.append(loss)
            correct += corr
            aux_losses.append(aux)
        _clearline()
        loss = float(np.mean(losses))
        accuracy = float(correct)/len(train_loader.dataset)
        aux_loss = float(np.mean(aux_losses))
        return loss, accuracy, aux_loss

    for epoch in range(start_epoch,max_epochs):
        # Training
        t0 = time.time()
        l,a,x = train_epoch()
        time_per_example = (time.time()-t0)/len(train_loader.dataset)
        stats['loss']['train'].append(l)
        stats['acc']['train'].append(a)
        stats['aux_loss']['train'].append(x)
        print(('Epoch %3i:    Training loss = %6.4f    accuracy = %6.4f'
              +'    aux loss = %6.4f    %4.2f msec/example')
              %(epoch,l,a,x,time_per_example*1000))

        # Validation
        t0 = time.time()
        l,a,x = val_epoch()
        time_per_example = (time.time()-t0)/len(val_loader.dataset)
        stats['loss']['val'].append(l)
        stats['acc']['val'].append(a)
        stats['aux_loss']['val'].append(x)
        print(('            Validation loss = %6.4f    accuracy = %6.4f'
              +'    aux loss = %6.4f    %4.2f msec/example')
              %(l,a,x,time_per_example*1000))

        # Save results and update plots
        with open(statsfile,'w') as sf:
            json.dump(stats,sf)
        plot_stats(stats,exp_path)

        # Early stopping
        if l<best_val:
            best_val = l
            stall = 0
            torch.save(model,os.path.join(exp_path,'checkpoint'))
        else:
            stall += 1
        if stall>=patience:
            break






