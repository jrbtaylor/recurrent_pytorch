"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import argparse
import json
import os

import torch

import data
import model
import train
from post import visualizations


MODELS_IMPLEMENTED = ['rnn','gru','surprise_gru']


def run(batch_size,permuted,modeltype='surprise_gru',n_hidden=64,zoneout=0.25,
        layer_norm=True,optimizer='adam',learnrate=1e-3,aux_weight=0.1,
        cuda=True,resume=False):
    assert isinstance(batch_size,int)
    assert isinstance(permuted,bool)
    assert modeltype in MODELS_IMPLEMENTED
    assert isinstance(n_hidden,int)
    assert isinstance(zoneout,(int,float))
    assert isinstance(layer_norm,bool)
    assert isinstance(optimizer,str)
    assert isinstance(learnrate,(int,float))
    assert isinstance(cuda,bool)
    assert isinstance(resume,bool)

    # Name the experiment s.t. parameters are easily readable
    exp_name = ('%s_perm%r_h%i_z%2f_norm%r_%s'
                %(modeltype,permuted,n_hidden,zoneout,layer_norm,optimizer))
    exp_path = os.path.join('/home/jason/experiments/recurrent_pytorch/',
                            exp_name)
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    if not resume:
        # Store experiment params in params.json
        params = {'batch_size':batch_size,'permuted':permuted,
                  'modeltype':modeltype,'n_hidden':n_hidden,'zoneout':zoneout,
                  'layer_norm':layer_norm,'optimizer':optimizer,
                  'learnrate':learnrate,'aux_weight':aux_weight,'cuda':cuda}
        with open(os.path.join(exp_path,'params.json'),'w') as f:
            json.dump(params,f)

        # Model
        if modeltype.lower()=='rnn':
            net = model.RNN(1,n_hidden,10,layer_norm)
        elif modeltype.lower()=='gru':
            net = model.GRU(1,n_hidden,10,layer_norm)
        elif modeltype.lower()=='surprise_gru':
            net = model.SurpriseGRU(1,n_hidden,10,layer_norm)
        else:
            raise ValueError
    else:
        # if resuming, need to have params, stats and checkpoint files
        if not (os.path.isfile(os.path.join(exp_path,'params.json'))
                and os.path.isfile(os.path.join(exp_path,'stats.json'))
                and os.path.isfile(os.path.join(exp_path, 'checkpoint'))):
            raise Exception('Missing params, stats or checkpoint file (resume)')
        net = torch.load(os.path.join(exp_path,'checkpoint'))

    # Data loaders
    train_loader, val_loader = data.mnist(batch_size, sequential=True,
                                          permuted=permuted)

    # Train
    train.fit_recurrent(train_loader,val_loader,net,exp_path,zoneout,optimizer,
                        aux_weight=aux_weight,cuda=cuda,resume=resume)

    # Post-trainign visualization
    post_training(exp_path,val_loader)


def post_training(modeldir,val_loader=None):
    if val_loader is None:
        with open(os.path.join(modeldir,'params.json'),'r') as jsf:
            params = json.load(jsf)
        _,val_loader = data.mnist(1,sequential=True,permuted=params['permuted'])
    visualizations(modeldir,val_loader)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Train recurrent net on sequential (or permuted) MNIST')
    parser.add_argument('--modeltype',
                        choices=MODELS_IMPLEMENTED,
                        default='surprise_gru')
    parser.add_argument('--zoneout', type=float, default=0.25)
    parser.add_argument('--permuted', action='store_true')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--aux_weight', type=float, default=0.1)
    parser.add_argument('--optimizer', choices=['adam','sgd','adamax'],
                        default='adam')
    parser.add_argument('--learnrate', type=float, default=0.001)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    run(args.batch_size,args.permuted,args.modeltype,args.n_hidden,args.zoneout,
        args.layer_norm,args.optimizer,args.learnrate,args.aux_weight,
        resume=args.resume)
