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
import pickle
from queue import PriorityQueue
import heapq as hq

import lmdb
import operator

# from torch.utils.tensorboard import SummaryWriter

#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = True

torch.manual_seed(0)
np.random.seed(0)   

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
from collections import defaultdict
import pickle
import random
import pandas as pd

import collections

class EPIC(Dataset):
    def __init__(self, lmdb_path, ann_file, args, transform=None):
        self.feat_list = []
        self.verbs_list = []
        self.nouns_list = []
        self.actions_list = []
        self.uids_list = []
        self.discarded_ids = []
        self.sampling = args.sampling
        self.challenge = args.challenge
        self.validation = args.validation
        count_empty_segs = 0
        obs_sec = args.obs_sec
        ant_sec = args.ant_sec
        fps = 30 # for egtea, ek100 and ek55
        if args.dataset == 'egtea':
            num_classes = {'verb': 19 , 'noun': 51, 'act': 106}
        if args.dataset == 'ek55':
            num_classes = {'verb': 125, 'noun': 352}
        if args.dataset == 'ek100':
            num_classes = {'verb': 97, 'noun': 300}
       
        action_annotation = pd.read_csv(ann_file, header=None)
        # [VID, Obs_start, Obs_end, Obs_noun, Obs_verb, Fut_start, Fut_end, Fut_noun,Fut_verb]
        env = []
        if args.late_fusion or self.challenge or self.validation:
            for path, mode in zip(lmdb_path, args.modality):
                if mode != 'vit':
                    env.append(lmdb.open(path, readonly=True, lock=False))
        else:
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        
        video_ids = list(action_annotation[1].unique())
        video_count = 0
        for video_id in video_ids[:]:
            video_count += 1
            if args.debug_on:
                if video_count > 2:
                    break
            if 'vit' in args.modality:
                if args.late_fusion or self.challenge or self.validation:
                    vit_lmdb_path = os.path.join(lmdb_path[-1], video_id.split()[0]+'.MP4')    
                    vit_env = lmdb.open(vit_lmdb_path, readonly=True, lock=False)
                    env.append(vit_env)
                else:
                    vit_lmdb_path = os.path.join(lmdb_path, video_id.split()[0]+'.MP4')    
                    env = lmdb.open(vit_lmdb_path, readonly=True, lock=False)
                    
            video_id = video_id.strip('\n')                
            if self.challenge:
                starts = list(action_annotation.loc[action_annotation[1] == video_id][2].values)[:]
                stops = list(action_annotation.loc[action_annotation[1] == video_id][3].values)[:]
                labels = list(action_annotation.loc[action_annotation[1] == video_id][0].values)[:]
                # actually uids
            else:
                starts = list(action_annotation.loc[action_annotation[1] == video_id][2].values)[1:]
                stops = list(action_annotation.loc[action_annotation[1] == video_id][3].values)[1:]
                cur_verbs = list(action_annotation.loc[action_annotation[1] == video_id][4].values)[:-1]
                next_verbs = list(action_annotation.loc[action_annotation[1] == video_id][4].values)[1:]
                cur_nouns = list(action_annotation.loc[action_annotation[1] == video_id][5].values)[:-1]
                next_nouns = list(action_annotation.loc[action_annotation[1] == video_id][5].values)[1:]
                cur_actions = list(action_annotation.loc[action_annotation[1] == video_id][6].values)[:-1]
                next_actions = list(action_annotation.loc[action_annotation[1] == video_id][6].values)[1:]
                
                verbs = zip(cur_verbs, next_verbs)
                nouns = zip(cur_nouns, next_nouns)
                actions = zip(cur_actions, next_actions)
                labels = zip(verbs, nouns, actions)
                
                
                freq_verbs = [0.0]*num_classes['verb']
                freq_nouns = [0.0]*num_classes['noun']
                
                dict_freq_verbs = collections.Counter(next_verbs) 
                for key in dict_freq_verbs.keys():
                    freq_verbs[key] = dict_freq_verbs[key]/len(next_verbs)
                dict_freq_nouns = collections.Counter(next_nouns)
                for key in dict_freq_nouns.keys():
                    freq_nouns[key] = dict_freq_nouns[key]/len(next_nouns)
                
                self.freq_info = {}
                
                self.freq_info['verb'] = freq_verbs
                self.freq_info['noun'] = freq_nouns
            #print(stops[-1])
            
            # print(feat.shape)
            video_id = video_id.split()[0]
            
            feat_in_video = []
            feat_in_video_verb = []
            feat_in_video_noun = []
            verbs_in_video = []
            nouns_in_video = []
            actions_in_video = []
            uids_in_video = []
            for start, stop, label in zip(starts, stops, labels):
                # if verb[0] in many_verbs and verb[1] in many_verbs:
                feat_in_seg = []
                # stop = stop - 30
                # if stop - start > seg_length:
                if self.challenge:
                    uid = label
                else:
                    verb, noun, action = label
                if args.dataset == 'ek55' or args.dataset == 'ek100' or args.dataset == 'egtea':
                    new_stop = stop
                    new_start = stop - int(fps*ant_sec) - int(fps*obs_sec)
                    if new_start < 1:
                        new_start = 1
                else:
                    new_stop = start
                    new_start = start - 30*obs_sec
                    if new_start < 1:
                        new_start = 1

                num_feats_seg = 0
                    
                for i in range(int(new_start),int(new_stop)):
                    # 'P24_03_frame_0000000578.jpg'
                    frame_num = video_id+'_frame_'+str(i).zfill(10)+'.jpg'
                    if isinstance(env, list):
                        ff = []
                        for e in env:
                            with e.begin() as feats:
                                ff.append(feats.get(frame_num.encode('utf-8')))
                        if not any(f is None for f in ff):
                            feats = []
                            num_feats_seg += 1
                            for f in ff:
                                feat = np.frombuffer(f, 'float32')
                                feats.append(feat.copy())
                            feat_in_seg.append(feats)
                            # print(len(feat_in_seg))
                    else:
                        with env.begin() as feats: 
                            f = feats.get(frame_num.encode('utf-8'))
                            if f is not None:
                                num_feats_seg += 1
                                feat = np.frombuffer(f, 'float32')
                            feat_in_seg.append(feat.copy())
                if num_feats_seg > 1:
                    feat_in_video.append(feat_in_seg)
                    if self.challenge:
                        uids_in_video.append(uid)
                    else:
                        verbs_in_video.append(verb)
                        nouns_in_video.append(noun)
                        actions_in_video.append(action)
                elif num_feats_seg == 1:
                    print('num_feats_seg:', num_feats_seg)
                else:   
                    if self.challenge:
                        self.discarded_ids.append(uid)
                    count_empty_segs += 1
            if len(feat_in_video) != 0:
                self.feat_list.extend(feat_in_video)
                if args.challenge:
                    self.uids_list.extend(uids_in_video)
                else:
                    self.verbs_list.extend(verbs_in_video)
                    self.nouns_list.extend(nouns_in_video) 
                    self.actions_list.extend(actions_in_video)
            else:
                print(video_id)
        print(count_empty_segs)         
        
    def __getitem__(self, index):

        feat_seq = self.feat_list[index][::self.sampling]
        
        if self.challenge:
            uid = self.uids_list[index]
            return feat_seq, uid
        else:
            action = self.actions_list[index]
            verb = self.verbs_list[index]
            noun = self.nouns_list[index]
            return feat_seq, action, verb, noun

    def __len__(self):
        return len(self.feat_list)
