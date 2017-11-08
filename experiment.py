"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import argparse

import data
import model
import train

def run(batch_size,permuted,modeltype='rnn',n_hidden=64,layer_norm=True,
        optimizer='adam',learnrate=1e-3,cuda=True):
    train_loader,val_loader = data.mnist(batch_size,sequential=True,
                                         permuted=permuted)
    if modeltype.lower()=='rnn':
        net = model.RNN(1,n_hidden,10,layer_norm)
    else:
        raise ValueError

    train.fit_recurrent(train_loader,val_loader,net,optimizer,
                        cuda=cuda)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Train recurrent net on sequential (or permuted) MNIST')
    parser.add_argument('--modeltype',
                        choices=['rnn'],
                        default='rnn')
    parser.add_argument('--permuted', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=16)
    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--optimizer', choices=['adam','sgd','adamax'],
                        default='adam')
    parser.add_argument('--learnrate', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true')

    args = parser.parse_args()
    run(args.batch_size,args.permuted,args.modeltype,args.n_hidden,
        args.layer_norm,args.optimizer,args.learnrate,args.cuda)
