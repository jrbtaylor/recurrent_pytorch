"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import json
import os
from PIL import Image, ImageDraw, ImageFont

import imageio
import numpy as np
import progressbar
from skimage.transform import resize
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib
# Disable Xwindows backend before importing matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data

PERMUTATION = data.pixel_permutation().numpy()
UNPERMUTATION = np.argsort(PERMUTATION)
IMAGE_SIZE = [28,28]


def _clearline():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)


def visualizations(modeldir,val_loader,loss_fcn=nn.NLLLoss(),
                   n_collage=100,n_gif=10):
    """
    Post-training visualizations:
    1. Visualize backprop-to-input (static)
    2. Save collages of lowest and highest-loss images with backprop-to-input
    3. Make gif
    """
    n_vis = max(n_collage,n_gif)

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
        def reshape_img(x):
            x = x[UNPERMUTATION]
            return np.reshape(x,IMAGE_SIZE)
    else:
        def reshape_img(x):
            return np.reshape(x,IMAGE_SIZE)
    def overlay_grad(x,gx):
        x, gx = np.squeeze(x), np.squeeze(gx)
        # absolute value, normalize, contrast stretch and threshold
        gx = np.abs(gx)
        gx = (gx-np.min(gx))/(np.max(gx)-np.min(gx)+1e-10)
        gx = gx**0.5
        gx = 0.5*gx*(gx>(np.mean(gx)-np.std(gx))) \
                +0.5*gx*(gx>(np.mean(gx)+np.std(gx)))
        x = reshape_img(x)
        gx = reshape_img(gx)
        overlay = np.transpose([0.8*x+0.2*gx,gx,gx],[1,2,0])
        # imageio.imwrite(os.path.join(savedir,str(file_id)+'.bmp'),overlay)
        return overlay

    # Run the validation set through the network
    print('Running validation...')
    def val_batch(x,y):
        model.dropout.p = 0
        hidden = model.init_hidden(x.data.size()[0])
        if cuda:
            hidden = hidden.cuda()
        outputs = []
        for i in range(x.size()[1]):
            output,h_new = model(x[:,i],hidden)
            # take expectation of zoneout
            hidden = zoneout*hidden+(1-zoneout)*h_new
            outputs.append(output.data.cpu().numpy())
        loss = loss_fcn(output,y).data.cpu().numpy()
        torch.max(output).backward()
        return loss, outputs, x.grad.data.cpu().numpy()

    # Find the best and worst n_vis examples
    best_imgs = [None]*n_vis
    best_grads = [None]*n_vis
    best_losses = [np.inf]*n_vis
    best_labels = [None]*n_vis
    best_outputs = [None]*n_vis
    replace_best = 0
    wrst_imgs = [None]*n_vis
    wrst_grads = [None]*n_vis
    wrst_losses = [-np.inf]*n_vis
    wrst_labels = [None]*n_vis
    wrst_outputs = [None]*n_vis
    replace_wrst = 0
    bar = progressbar.ProgressBar()
    # i = 0
    for x,y in bar(val_loader):
        # if i>n_vis:
        #     break
        # i += 1
        # print(i)
        if cuda:
            x,y = x.cuda(), y.cuda()
        x,y = Variable(x,requires_grad=True), Variable(y)
        loss,outputs,xgrad = val_batch(x,y)
        loss = loss[0]
        if loss<best_losses[replace_best]:
            best_losses[replace_best] = loss
            best_imgs[replace_best] = x.data.cpu().numpy()
            best_grads[replace_best] = xgrad
            best_labels[replace_best] = y.data.cpu().numpy()
            best_outputs[replace_best] = outputs
            replace_best = np.argmax(best_losses)
        if loss>wrst_losses[replace_wrst]:
            wrst_losses[replace_wrst] = loss
            wrst_imgs[replace_wrst] = x.data.cpu().numpy()
            wrst_grads[replace_wrst] = xgrad
            wrst_labels[replace_wrst] = y.data.cpu().numpy()
            wrst_outputs[replace_wrst] = outputs
            replace_wrst = np.argmin(wrst_losses)
    _clearline()

    # Convert outputs from logsoftmax to regular softmax
    best_outputs = [np.exp(o) for o in best_outputs]
    wrst_outputs = [np.exp(o) for o in wrst_outputs]

    # Make gifs of best and worst examples
    gifdir = os.path.join(modeldir,'gifs')
    if not os.path.isdir(gifdir):
        os.makedirs(gifdir)
    for i in range(n_gif):
        make_gif(best_imgs[i],best_outputs[i],best_labels[i],
                 os.path.join(gifdir,'best'+str(i)+'.gif'))
        make_gif(wrst_imgs[i], wrst_outputs[i], wrst_labels[i],
             os.path.join(gifdir, 'worst'+str(i)+'.gif'))

    # Make collages of best and worst examples
    def frmt(output):
        return ' '.join(['%.3f'%o for o in output[0].tolist()])
    imgs = [overlay_grad(best_imgs[i],best_grads[i]) for i in range(n_collage)]
    txt = ['Label: %i\nOutput: %s\nLoss: %.2f'
           %(best_labels[i],frmt(best_outputs[i][-1]),best_losses[i])
           for i in range(n_collage)]
    collage(imgs,os.path.join(modeldir,'best.jpeg'),txt,fontsize=10)
    imgs = [overlay_grad(wrst_imgs[i],wrst_grads[i]) for i in range(n_collage)]
    txt = ['Label: %i\nOutput: %s\nLoss: %.2f'
           %(wrst_labels[i],frmt(wrst_outputs[i][-1]),wrst_losses[i])
           for i in range(n_collage)]
    collage(imgs,os.path.join(modeldir,'worst.jpeg'),txt,fontsize=10)


def make_gif(img,output,label,saveto):
    # Show image appear one pixel at a time (left)
    # and the output from what it's seen so far (right)
    img = np.squeeze(img)
    img_gif = [100*np.ones(IMAGE_SIZE,dtype='uint8')]
    for t,pixel_idx in enumerate(PERMUTATION):
        r,c = pixel_idx//IMAGE_SIZE[0], pixel_idx%IMAGE_SIZE[0]
        frame = np.copy(img_gif[-1]).astype('uint8')
        frame[r,c] = np.squeeze(255*img[t]).astype('uint8')
        img_gif.append(frame)

    def bar_chart(out,lab):
        fig = plt.figure(num=1,figsize=(4,4),dpi=70,facecolor='w',edgecolor='k')
        x_rest = list(range(out.size))
        x_label = x_rest.pop(lab)
        y_rest = out.tolist()
        y_label = y_rest.pop(lab)
        _ = plt.bar(x_rest,y_rest,width=0.8,color='r')
        _ = plt.bar(x_label,y_label,width=0.8,color='g')
        plt.title('Outputs')
        plt.rcParams.update({'font.size':10})
        fig.tight_layout(pad=0)
        fig.canvas.draw()

        # now convert the plot to a numpy array
        plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot = plot.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        plt.close(fig)
        return plot

    plot_gif = []
    for i in range(output.shape[0]):
        plot = bar_chart(np.squeeze(output[i]),np.squeeze(label))
        plot_gif.append(plot.astype('uint8'))

    # combine the image and plot
    combined = []
    for i,p in zip(img_gif,plot_gif):
        if i.ndim==2:
            i = np.repeat(i[:,:,np.newaxis],3,axis=2)
        i = resize(i,[p.shape[0],int(i.shape[1]*p.shape[0]/i.shape[0])],order=0,
                   preserve_range=True,mode='constant').astype('uint8')
        combined.append(np.concatenate((i,p),axis=1))

    imageio.mimsave(saveto,combined,format='gif',loop=0,
                    duration=[0.005]*(len(combined)-1)+[1])


def collage(imgs,saveto,text=None,fontsize=10):
    """
    Save a collage of images (e.g. to see highest vs lowest-loss examples)
    """
    assert isinstance(imgs,list)

    # assume >4:3 aspect ratio, figure out # of columns
    columns = np.ceil(np.sqrt(len(imgs)*4/3))
    while len(imgs)%columns!=0:
        columns += 1

    has_text = text is not None
    if has_text:
        assert isinstance(text,list)
        assert all([type(t) is str for t in text])
        assert len(imgs)==len(text)
        lines = np.max([t.count('\n') for t in text])+1
    else:
        text = ['']*len(imgs)

    row = []
    result = []
    for idx,(img,t) in enumerate(zip(imgs,text)):
        img = resize(img,[280,280],order=0,preserve_range=True,mode='constant')
        img = np.array(255*img, dtype='uint8')
        img_size = img.shape
        if img.ndim==2:
            img = np.repeat(img[:,:,np.newaxis],3,axis=2)
        if has_text:
            img = np.concatenate((img,np.zeros((int(1.5*lines*fontsize),
                                                img_size[1],3),dtype='uint8')))
            img = Image.fromarray(img,'RGB')
            imgdraw = ImageDraw.Draw(img)
            imgdraw.text((0,img_size[0]),t,(255,255,255),
                         font=ImageFont.truetype('UbuntuMono-R.ttf',fontsize))
            img = np.array(img, dtype='uint8')
        row.append(img)
        if (idx+1)%columns==0:
            result.append(row)
            row = []
    result = np.array(result)
    result = np.concatenate(result,axis=1)
    result = np.concatenate(result,axis=1)
    Image.fromarray(result).save(saveto)