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
from scipy.special import softmax
import sys
sys.path.append(".")
import argparse
import json
from tqdm import tqdm
#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)


from collections import defaultdict
import pickle
import random
import pandas as pd

from dataset import EPIC
from models import VarAnt
from training import TrainTest
from utils import topk_accuracy, get_top5actions

def combine_verb_noun_preds(res_verb, res_noun):
    """
    Args:
        res_verb (matrix with NxC1 dims)
        res_noun (matrix with NxC2 dims)
    Returns:
        res_action (matrix with Nx(C1 * C2) dims)
    """
    num_elts = res_verb.shape[0]
    # normalize the predictions using softmax
    res_verb = softmax(res_verb, axis=-1)
    res_noun = softmax(res_noun, axis=-1)
    # Cross product to get the combined score
    return np.einsum('ij,ik->ijk', res_verb, res_noun).reshape((num_elts, -1))
        
   
def validate(test_set, anticipation_model, num_classes, args):
    testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    preds = []
    targets = []
    for i, data in tqdm(enumerate(testloader, 0), total=len(test_set)): 
        # seq_segs, uid = data 
        seq_segs, action, verb, noun = data
        if args.late_fusion:
            scores = latefusioneval(seq_segs, anticipation_model, num_classes, args.weights)
        else:
            feat_num = 0
            scores = singleevalmodified(seq_segs, anticipation_model, feat_num)
        
        preds.append(scores)
        if args.outputs[0] == 'verb':
            targets.append(verb[1])
        elif args.outputs[0] == 'noun':
            targets.append(noun[1])
            
    preds = torch.stack(preds)
    print(preds.shape)
    targets = torch.stack(targets)
    TEST_ACC = []
    TEST_ACC.append(topk_accuracy(preds, targets, k=1).item()*100)
    TEST_ACC.append(topk_accuracy(preds, targets, k=5).item()*100) 
    print('[{:s}@1 {:.2f}], [{:s}@5 {:.2f}]'.format(\
                       args.outputs[0], TEST_ACC[0], args.outputs[0], TEST_ACC[1]))
    
       

            
def challengeeval(test_set, json_file, verb_anticipation_model, noun_anticipation_model, verb_num_classes, noun_num_classes, args):
    predictions = {}
    if args.dataset == 'ek55':
        predictions = {'version': '0.1',\
                  'challenge': 'action_anticipation', 'results': {}}
        possible_actions = pd.read_csv('/data/roy/graph/EPIC_KITCHENS_2020/actions.csv',index_col='id')
    if args.dataset == 'ek100':
        predictions = {'version': '0.2',\
                  'challenge': 'action_anticipation', 'results': {}}
        predictions['sls_pt'] = 1
        predictions['sls_tl'] = 4
        predictions['sls_td'] = 3
        possible_actions = pd.read_csv('/data/roy/graph/EPIC_KITCHENS_2020/EPIC_100_actions.csv',index_col='id')
    
    testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    
    softmax = nn.Softmax(dim=1)
           
    for i, data in tqdm(enumerate(testloader, 0), total=len(test_set)): 
        # seq_segs, uid = data 
        seq_segs, uid = data
        pred = []
        # print('seq_segs[1]', seq_segs[1])
        if args.verb_fusion:
            verb_scores = latefusioneval(seq_segs, verb_anticipation_model, verb_num_classes, args.verb_weights)
        else:
            feat_num = 0 # feature for verb is supplied first
            verb_scores = singleevalmodified(seq_segs, verb_anticipation_model, feat_num)
        
        if args.noun_fusion:
            noun_scores = latefusioneval(seq_segs, noun_anticipation_model, noun_num_classes, args.noun_weights)
        else:
            feat_num = 1 # feature for noun is supplied second
            noun_scores = singleevalmodified(seq_segs, noun_anticipation_model, feat_num)
            
        # uid = i
        if args.dataset == 'ek55':
            uid = uid.item()    
        else:
            uid = uid[0]
        predictions['results'][str(uid)] = {}
        predictions['results'][str(uid)]['verb'] = {str(ii): round(float(verb_scores[0,ii]),5) for ii in range(verb_num_classes)}
        predictions['results'][str(uid)]['noun'] = {str(ii): round(float(noun_scores[0,ii]),5) for ii in range(noun_num_classes)}
            # predictions['results'][str(uid)]['action'] = {str(v)+','+str(n): round(float(action_scores[v,n]),5) for v,n in top100_actions}      
    for uid in test_set.discarded_ids:
        if args.dataset == 'ek55':
            uid = uid.item()
        else:
            uid = uid[0]
        predictions['results'][str(uid)] = {}
        predictions['results'][str(uid)]['verb'] = {str(ii): 0.0 for ii in range(verb_num_classes)}
        predictions['results'][str(uid)]['noun'] = {str(ii): 0.0 for ii in range(noun_num_classes)}
        
    with open(json_file, 'w') as fp:
        json.dump(predictions, fp,  indent=4) 


def latefusioneval(seq_segs, anticipation_models, num_classes, weights):
    softmax = nn.Softmax(dim=1)
    pred_total = torch.zeros(1,num_classes) # for verb
    for j in range(len(anticipation_models)):
        feat_seq = []
        for seq in seq_segs: 
            feat_seq.append(seq[j])
        feat_seq = torch.stack(feat_seq).float().squeeze(0).to(device)
        
        pred, _, _, _, _ = anticipation_models[j](feat_seq)
        # print(softmax(pred_verb.detach().cpu())[0])
        pred_total += torch.mul(softmax(pred.detach().cpu()),float(weights[j]))

    return pred_total
    
def singleeval(seq_segs, anticipation_model):
    softmax = nn.Softmax(dim=1)
    preds = []
    targets = []
    feat_seq = []
    for seq in seq_segs: 
        feat_seq.append(seq)
    feat_seq = torch.stack(feat_seq).float().squeeze(0).to(device)
        
    pred, _, _, _, _ = anticipation_model(feat_seq)
    # print(softmax(pred_verb.detach().cpu())[0])
    pred = softmax(pred.detach().cpu())

    return pred   

def singleevalmodified(seq_segs, anticipation_model, feat_num):
    softmax = nn.Softmax(dim=1)
    preds = []
    targets = []
    feat_seq = []
    for seq in seq_segs: 
        feat_seq.append(seq[feat_num])
    feat_seq = torch.stack(feat_seq).float().squeeze(0).to(device)
        
    pred, _, _, _, _ = anticipation_model(feat_seq)
    # print(softmax(pred_verb.detach().cpu())[0])
    pred = softmax(pred.detach().cpu())

    return pred      