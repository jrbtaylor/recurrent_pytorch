"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import argparse
import json
import os

import data
import model
import train

MODELS_IMPLEMENTED = ['rnn','gru']

def run(batch_size,permuted,modeltype='gru',n_hidden=64,zoneout=0.25,
        layer_norm=True,optimizer='adam',learnrate=1e-3,cuda=True):
    assert isinstance(batch_size,int)
    assert isinstance(permuted,bool)
    assert modeltype in MODELS_IMPLEMENTED
    assert isinstance(n_hidden,int)
    assert isinstance(zoneout,(int,float))
    assert isinstance(layer_norm,bool)
    assert isinstance(optimizer,str)
    assert isinstance(learnrate,(int,float))
    assert isinstance(cuda,bool)

    # Name the experiment s.t. parameters are easily readable
    exp_name = ('%s_perm%r_h%i_z%2f_norm%r_%s'
                %(modeltype,permuted,n_hidden,zoneout,layer_norm,optimizer))
    exp_path = os.path.join('/home/jason/experiments/recurrent_pytorch/',
                            exp_name)
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    # Store experiment params in params.json
    params = {'batch_size':batch_size,'permuted':permuted,'modeltype':modeltype,
              'n_hidden':n_hidden,'zoneout':zoneout,'layer_norm':layer_norm,
              'optimizer':optimizer,'learnrate':learnrate,'cuda':cuda}
    with open(os.path.join(exp_path,'params.json'),'w') as f:
        json.dump(params,f)

    # Data and model
    train_loader,val_loader = data.mnist(batch_size,sequential=True,
                                         permuted=permuted)
    if modeltype.lower()=='rnn':
        net = model.RNN(1,n_hidden,10,layer_norm)
    elif modeltype.lower()=='gru':
        net = model.GRU(1,n_hidden,10,layer_norm)
    else:
        raise ValueError

    # Train
    train.fit_recurrent(train_loader,val_loader,net,exp_path,
                        zoneout,optimizer,cuda=cuda)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Train recurrent net on sequential (or permuted) MNIST')
    parser.add_argument('--modeltype',
                        choices=MODELS_IMPLEMENTED,
                        default='gru')
    parser.add_argument('--zoneout', type=float, default=0.25)
    parser.add_argument('--permuted', action='store_true')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--optimizer', choices=['adam','sgd','adamax'],
                        default='adam')
    parser.add_argument('--learnrate', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true')

    args = parser.parse_args()
    run(args.batch_size,args.permuted,args.modeltype,args.n_hidden,args.zoneout,
        args.layer_norm,args.optimizer,args.learnrate,args.cuda)
