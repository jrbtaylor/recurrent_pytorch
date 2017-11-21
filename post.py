"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import json
import os

import imageio
import numpy as np
import progressbar
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib
# Disable Xwindows backend before importing matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _clearline():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)


def visualizations(modeldir,val_loader,loss_fcn=nn.NLLLoss()):
    """
    Post-training visualizations:
    1. Visualize backprop-to-input (static)
    2. Save collages of lowest and highest-loss images with backprop-to-input
    3. Make gif
    """

    # Reload the saved net
    model = torch.load(os.path.join(modeldir,'checkpoint'))
    with open(os.path.join(modeldir,'params.json'),'r') as jsf:
        params = json.load(jsf)
    zoneout = params['zoneout']
    cuda = params['cuda']
    permuted = params['permuted']
    if cuda:
        model = model.cuda()

    # Make path for saving images
    savedir = os.path.join(modeldir,'input_grads')
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    # Function to reshape img and grad, overlay and save
    if permuted:
        raise ValueError # need to implement saving the pattern to re-use
    else:
        def reshape_img(x):
            return np.reshape(x,[28,28])
    def overlay_grad(x,gx,file_id):
        # absolute value, normalize, contrast stretch and threshold
        gx = np.abs(gx)
        gx = (gx-np.min(gx))/(np.max(gx)-np.min(gx)+1e-10)
        gx = gx**0.5
        gx = 0.5*gx*(gx>(np.mean(gx)-np.std(gx))) \
                +0.5*gx*(gx>(np.mean(gx)+np.std(gx)))
        x = reshape_img(x)
        gx = reshape_img(gx)
        overlay = np.transpose([0.8*x+0.2*gx,gx,gx],[1,2,0])
        imageio.imwrite(os.path.join(savedir,str(file_id)+'.bmp'),overlay)

    # Run the validation set through the network
    print('Running validation...')
    def val_batch(x,y):
        model.dropout.p = 0
        hidden = model.init_hidden(x.data.size()[0])
        if cuda:
            hidden = hidden.cuda()
        correct = 0
        for i in range(x.size()[1]):
            output,h_new = model(x[:,i],hidden)
            # take expectation of zoneout
            hidden = zoneout*hidden+(1-zoneout)*h_new
        loss = loss_fcn(output,y)
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        torch.max(output).backward()
        return loss.data.cpu().numpy(), correct, x.grad.data.cpu().numpy()

    bar = progressbar.ProgressBar()
    losses = []
    correct = 0
    file_id = 0
    for x,y in bar(val_loader):
        file_id += 1
        xnp = x.numpy()
        if cuda:
            x,y = x.cuda(), y.cuda()
        x,y = Variable(x,requires_grad=True), Variable(y)
        loss,corr,xgrad = val_batch(x,y)
        losses.append(loss)
        correct += corr
        for i in range(xnp.shape[0]):
            overlay_grad(xnp[i],xgrad[i],file_id)
        if file_id>=50:  # probably don't want to save too many
            break
    _clearline()

