#!/usr/bin/env python
# coding: utf-8
# author: Debaditya Roy

import time, os, copy, numpy as np

import torch, torchvision
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter, init
import torch.nn.functional as F
import sys
sys.path.append(".")

import argparse

# from torch.utils.tensorboard import SummaryWriter


#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


from collections import defaultdict
import pickle
import random
import pandas as pd


from dataset import EPIC
from models import VarAnt
from training import TrainTest
from evaluation import challengeeval, validate



parser = argparse.ArgumentParser(description='Variational Latent Goal model for EK55 and EK100', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--modality', nargs='+', default=None, choices=['rgb', 'flow', 'obj' ], help='Choose tsn (rgb or flow) or obj or a combination for fusion', required=True)
parser.add_argument('--dataset', type=str, default='ek55', choices=['ek55', 'ek100', 'egtea'], help='Choose between EK55, EK100 and EGTEA')
parser.add_argument('--outputs', nargs='+', choices=['verb', 'noun', 'action', 'act'], help='Choose between verb and noun or act for EGTEA')
parser.add_argument('--obs_sec',dest='obs_sec', type=int, default=2, choices=[1, 2, 3, 4, 5, 6], help='Choose observed duration in secs 1-6')
parser.add_argument('--ant_sec',dest='ant_sec', type=float, default=1, choices=[2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25], help='Choose anticipation time')
parser.add_argument('--latent_dim', type=int, default=128, help='Choose latent dimension')
parser.add_argument('--num_act_cand', type=int, default=10, choices=[1, 6, 10, 20, 30, 50, 100], help='Choose number of candidates to sample for next verb/noun- 10, 20, 30, 50, 100')
parser.add_argument('--num_goal_cand', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Choose number of abstract goals to sample- 1, 2, 3, 4, 5')
parser.add_argument('--hidden_dim', type=int, default=256, help='Choose hidden dimension')
parser.add_argument('--n_layers', type=int, default=1, choices=[1, 2, 3], help='Choose # layers in RNN for next visual feature - 1, 2, 3')
parser.add_argument('--dropout', type=float, default=0.8, help="Dropout rate")
parser.add_argument('--sampling', type=int, default=6, choices=[1, 2, 3, 5, 6, 10, 15], help='Choose sampling freq of input features, 2, 3, 5, 6, 10, 15')
parser.add_argument('--nepochs', type=int, default=10, help='Choose num epochs')
parser.add_argument('--scheduler', type=str, default='none', choices=['cosine', 'none'], help='Choose scheduler - cosineLR or none(AdamW)')
parser.add_argument('--batch_size', type=int, default=256, choices=[32, 64, 128, 256], help='Choose batch_size - 64, 128, 256')
parser.add_argument('--losses', nargs='+', choices=['og', 'na', 'ng', 'oa', 'gc', 'eqoa', 'eqna'], help='' )

# EGTEA parameter
parser.add_argument('--split', type=str, choices=['1', '2', '3', '4'], help='')


# validation parameters
parser.add_argument('--late_fusion', dest='late_fusion', action='store_true', help='Late fusion')
parser.add_argument('--fusion_weights', dest='weights', nargs='+', default=None, help='')
parser.add_argument('--scores', dest='scores', action='store_true', help='Choose Action scores or RGB Features')
parser.add_argument('--validate', dest='validation', action='store_true', help='')

# challenge parameters
parser.add_argument('--challenge', dest='challenge', action='store_true', help='Generate json on test set')

parser.add_argument('--verb_fusion', action='store_true', help='Late fusion for verb')
parser.add_argument('--verb_modes', nargs='+', choices=['rgb', 'flow', 'obj'], help='')
parser.add_argument('--verb_weights', nargs='+', help='')

parser.add_argument('--noun_fusion', action='store_true', help='Late fusion for noun')
parser.add_argument('--noun_modes', nargs='+', choices=['rgb', 'flow', 'obj'], help='')
parser.add_argument('--noun_weights', nargs='+', help='')
# Debugging True
parser.add_argument('--debug_on', action='store_true', help='')

# Equalization loss
parser.add_argument('--gamma', dest='gamma', help='Equalization Loss - gamma')
parser.add_argument('--lambda', dest='lambda_', help='Equalization Loss - lambda')

# Add validation to training set for EK100
parser.add_argument('--inc_val', action='store_true', help='')

args = parser.parse_args()

if args.dataset == 'ek55':
    num_classes = {'verb': 125, 'noun': 352}
    
    train_ann_file = '/data/roy/graph/EPIC_KITCHENS_2020/training.csv'
    val_ann_file = '/data/roy/graph/EPIC_KITCHENS_2020/validation.csv'
    test_ann_files = ['/data/roy/graph/EPIC_KITCHENS_2020/test_seen.csv', '/data/roy/graph/EPIC_KITCHENS_2020/test_unseen.csv'] 
    
    paths = { 'rgb': '/home/roy/epic_rgb_full_features', \
          'flow': '/data/Datasets/EPIC-KITCHENS_55/epic_flow_full_features', \
          'obj': '/home/roy/epic_bagofobj_full_features'}
    
    json_files = ['seen.json', 'unseen.json']
    
if args.dataset == 'ek100':
    num_classes = {'verb': 97, 'noun': 300}
    
    train_ann_file = '/data/roy/graph/EPIC_KITCHENS_2020/EPIC_100_training.csv'
    if args.inc_val:
        train_ann_file = '/data/roy/graph/EPIC_KITCHENS_2020/EPIC_100_training+validation.csv'
    val_ann_file = '/data/roy/graph/EPIC_KITCHENS_2020/EPIC_100_validation.csv'
    test_ann_files = ['/data/roy/graph/EPIC_KITCHENS_2020/EPIC_100_test_timestamps.csv']
    
    paths = { 'rgb': '/home/roy/epic100_tsnrgb_features', \
              'obj': '/home/roy/epic100_bagofobj_features',\
              'flow': '/data/Datasets/EPIC-KITCHENS_55/epic100_tsnflow_features'}    
    
    json_files = ['test.json']

if args.dataset == 'egtea':
    num_classes = {'verb': 19 , 'noun': 51, 'act': 106}
    
    train_ann_file = '../rulstm/RULSTM/data/egtea/training{:s}.csv'.format(args.split)
    val_ann_file = '../rulstm/RULSTM/data/egtea/validation{:s}.csv'.format(args.split)
        
    paths = { 'rgb': '/home/roy/EGTEA/TSN-C_3_egtea_action_CE_s{:s}_rgb_model_best_fcfull_hd'.format(args.split), \
              'flow': '/home/roy/EGTEA/TSN-C_3_egtea_action_CE_s{:s}_flow_model_best_fcfull_hd'.format(args.split)}    
 
    
dim_dict = {'rgb': 1024, 'flow': 1024, 'obj': 352}


if args.challenge: # Can take 2 or more inputs (late_fusion). Produces verb or noun
    verb_anticipation_model = []
    lmdb_paths = []
    if args.verb_fusion:
        for mode in args.modality:
            lmdb_paths.append(paths[mode])
            args.hidden_dim = 256
            output = 'verb'
            model = VarAnt(num_classes['verb'], args=args, feat_dim=dim_dict[mode])
            mode_ckpt = 'ckpt/{:s}_{:s}_{:s}_o_{:d}_a_{:.2f}_h_{:d}_z_{:d}_actcand_{:d}_goalcand_{:d}_samp_{:d}_sched_{:s}'\
                        .format(mode, args.dataset, putput, args.obs_sec, args.ant_sec, args.hidden_dim, args.latent_dim, \
                            args.num_act_cand, args.num_goal_cand, args.sampling, args.scheduler)
            mode_ckpt = mode_ckpt+'_'+'_'.join(args.losses)+'.pt'
            print(mode_ckpt)
            state = torch.load(mode_ckpt)
            model.load_state_dict(state['model'])
            model.to(device)
            model.eval()
            verb_anticipation_model.append(model)
    else:
        mode = args.modality[0]
        lmdb_paths.append(paths[mode])
        # args.hidden_dim = 256
        output = 'verb'
        verb_anticipation_model = VarAnt(num_classes['verb'], args=args, feat_dim=dim_dict[mode])
        
        verb_ckpt_path = 'ckpt/{:s}_{:s}_{:s}_o_{:d}_a_{:.2f}_h_{:d}_z_{:d}_actcand_{:d}_goalcand_{:d}_samp_{:d}_sched_{:s}'\
                        .format(mode, args.dataset, output, args.obs_sec, args.ant_sec, args.hidden_dim, args.latent_dim, \
                            args.num_act_cand, args.num_goal_cand, args.sampling, args.scheduler)
        verb_ckpt_path = verb_ckpt_path+'_'+'_'.join(args.losses)+'.pt'
        print(verb_ckpt_path)
        verb_state = torch.load(verb_ckpt_path)
        verb_anticipation_model.load_state_dict(verb_state['model'])
        verb_anticipation_model.to(device)    
        verb_anticipation_model.eval()
    
    noun_anticipation_model = []
    if args.noun_fusion:
        for mode in args.modality:
            args.hidden_dim = 1024
            output = 'noun'
            model = VarAnt(num_classes['noun'], args=args, feat_dim=dim_dict[mode])
            mode_ckpt = 'ckpt/{:s}_{:s}_{:s}_o_{:d}_a_{:.2f}_h_{:d}_z_{:d}_actcand_{:d}_goalcand_{:d}_samp_{:d}_sched_{:s}'\
                    .format(mode, args.dataset, output, args.obs_sec, args.ant_sec, args.hidden_dim, args.latent_dim, \
                            args.num_act_cand, args.num_goal_cand, args.sampling, args.scheduler)
            mode_ckpt = mode_ckpt+'_'+'_'.join(args.losses)+'.pt'
            print(mode_ckpt)
            state = torch.load(mode_ckpt)
            model.load_state_dict(state['model'])
            model.to(device)
            model.eval()
            noun_anticipation_model.append(model)
    else:
        mode = args.modality[1]
        # args.hidden_dim = 1024
        output = 'noun'
        lmdb_paths.append(paths[mode])
        noun_anticipation_model = VarAnt(num_classes['noun'], args=args, feat_dim=dim_dict[mode])
        
        noun_ckpt_path = 'ckpt/{:s}_{:s}_{:s}_o_{:d}_a_{:.2f}_h_{:d}_z_{:d}_actcand_{:d}_goalcand_{:d}_samp_{:d}_sched_{:s}'\
                    .format(mode, args.dataset, output, args.obs_sec, args.ant_sec, args.hidden_dim, args.latent_dim, \
                            args.num_act_cand, args.num_goal_cand, args.sampling, args.scheduler)
        noun_ckpt_path = noun_ckpt_path+'_'+'_'.join(args.losses)+'.pt'
        print(noun_ckpt_path)
        noun_state = torch.load(noun_ckpt_path)
        noun_anticipation_model.load_state_dict(noun_state['model'])
        noun_anticipation_model.to(device)    
        noun_anticipation_model.eval()
    
    for test_ann_file, json_file in zip(test_ann_files, json_files):
        test_set = EPIC(lmdb_paths, test_ann_file, args)
        print('{} test instances.'.format(len(test_set)))
        challengeeval(test_set, json_file, verb_anticipation_model, noun_anticipation_model, num_classes['verb'], num_classes['noun'], args)

elif args.validation:  # Takes in 1 input or more inputs (late_fusion). Produces verb or noun
    anticipation_model = []
    lmdb_paths = []
    num_cls = num_classes[args.outputs[0]]
    if args.late_fusion:
        for mode in args.modality:
            lmdb_paths.append(paths[mode])
            model = VarAnt(num_cls, args=args, feat_dim=dim_dict[mode])
            ckpt_path = 'ckpt/{:s}_{:s}_{:s}_o_{:d}_a_{:.2f}_h_{:d}_z_{:d}_actcand_{:d}_goalcand_{:d}_samp_{:d}_sched_{:s}'\
                    .format(mode, args.dataset, args.outputs[0], args.obs_sec, args.ant_sec, args.hidden_dim, args.latent_dim, \
                            args.num_act_cand, args.num_goal_cand, args.sampling, args.scheduler)
            ckpt_path = ckpt_path+'_'+'_'.join(args.losses)+'.pt'
            print(ckpt_path)
            state = torch.load(ckpt_path)
            model.load_state_dict(state['model'])
            model.to(device)    
            model.eval()
            anticipation_model.append(model)
    else:
        mode = args.modality[0]
        lmdb_paths.append(paths[mode])
        model = VarAnt(num_cls, args=args, feat_dim=dim_dict[mode])
        ckpt_path = 'ckpt/{:s}_{:s}_{:s}_o_{:d}_a_{:.2f}_h_{:d}_z_{:d}_actcand_{:d}_goalcand_{:d}_samp_{:d}_sched_{:s}'\
                    .format(mode, args.dataset, args.outputs[0], args.obs_sec, args.ant_sec, args.hidden_dim, args.latent_dim, \
                            args.num_act_cand, args.num_goal_cand, args.sampling, args.scheduler)
        ckpt_path = ckpt_path+'_'+'_'.join(args.losses)+'.pt'
        print(ckpt_path)
        state = torch.load(ckpt_path)
        model.load_state_dict(state['model'])
        model.to(device)    
        model.eval()
        anticipation_model = model
    
    val_set = EPIC(lmdb_paths, val_ann_file, args)
    print('{} val instances.'.format(len(val_set)))
    validate(val_set, anticipation_model, num_cls, args)
   
else: # training - only takes one modality and one output type - noun or verb
    mode = args.modality[0]
    output = args.outputs[0]
    args.num_classes = num_classes[output]
    lmdb_path = paths[mode]
    model_name = '{:s}_{:s}_{:s}_o_{:d}_a_{:.2f}_h_{:d}_z_{:d}_actcand_{:d}_goalcand_{:d}_samp_{:d}_sched_{:s}'\
                 .format(mode, args.dataset, output, args.obs_sec, args.ant_sec, args.hidden_dim, args.latent_dim, \
                         args.num_act_cand, args.num_goal_cand, args.sampling, args.scheduler)
    if args.split:
        model_name = '{:s}_split_{:s}'.format(model_name, args.split)
            
    model_name = model_name+'_'+'_'.join(args.losses)
    if args.inc_val:
        model_name = model_name+'_inc_val'
    if 'eqna' in args.losses:
        model_name = model_name+'_gamma_{:s}'.format(args.gamma)
        if args.lambda_:
            model_name = model_name+'_lambda_{:s}'.format(args.lambda_)
    ckpt_path = 'ckpt/{:s}.pt'.format(model_name)
    #writer = SummaryWriter(log_dir='logs/{:s}'.format(model_name))
    writer = None
    
    
    if args.dataset == 'ek55' or args.dataset == 'ek100' or args.dataset == 'egtea':
        training_set = EPIC(lmdb_path, train_ann_file, args)
        print('{} train instances.'.format(len(training_set)))
        val_set = EPIC(lmdb_path, val_ann_file, args)
        print('{} test instances.'.format(len(val_set)))
        anticipation_model = VarAnt(num_classes[output], args=args, feat_dim=dim_dict[mode])
        EXEC = TrainTest(anticipation_model, training_set, val_set, ckpt_path, writer, args)
        EXEC.train()
