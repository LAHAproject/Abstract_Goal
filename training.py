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

from utils import topk_accuracy, convert_to_action
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


from utils import plot_confusion, get_top5actions, convert_to_action

class TrainTest():
    def __init__(self, model, trainset, testset, ckpt_path, writer, args, fusion=False):
        self.model = model
        self.model.to(device)    

        self.trainset_size = len(trainset)
        self.fusion = fusion
        self.losses = args.losses
        
        self.dataset = args.dataset
        if args.scheduler == 'cosine':
            self.optimizer = optim.Adam(self.model.parameters())
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.trainset_size)
        else:        
            self.optimizer = optim.AdamW(self.model.parameters())
            self.scheduler = None
        # self.writer = writer
        #self.optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
        self.celoss = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()
        self.mmloss = nn.MarginRankingLoss()
        self.l1_crit = nn.L1Loss()
        
        self.num_classes = args.num_classes
        if args.lambda_:
            self.lambda_ = float(args.lambda_)
        if args.gamma:
            self.gamma = float(args.gamma)
        self.freq_info = torch.FloatTensor(trainset.freq_info[args.outputs[0]]).to(device)
        # print(self.freq_info)
        self.batch_size = args.batch_size
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=0)
          
        self.chkpath = ckpt_path
        self.batch_size = args.batch_size
        self.nepoch = args.nepochs

        self.output = args.outputs[0]

        if not os.path.exists('ckpt/'):
            os.mkdir('ckpt/')            
        print(self.chkpath)
        if os.path.exists(self.chkpath) == True:
            print('load from ckpt', end=' ')
            self.state = torch.load(self.chkpath)
            self.model.load_state_dict(self.state['model'])
            best_acc = self.state['acc']
            start_epoch = self.state['epoch']
            print('Epoch {}'.format(start_epoch))
            if start_epoch >= self.nepoch:
                print('testing as epoch is max.')
                TEST_ACC = self.test()
                if self.output == 'action':
                    print('V@1/test:', TEST_ACC[0])
                    print('V@5/test:', TEST_ACC[1])
                    print('N@1/test:', TEST_ACC[2])
                    print('N@5/test:', TEST_ACC[3])
                    print('A@1/test:', TEST_ACC[4])
                    print('A@5/test:', TEST_ACC[5])
                elif self.output == 'noun':
                    print('N@1/test:', TEST_ACC[0])
                    print('N@5/test:', TEST_ACC[1])
                elif self.output == 'verb':
                    print('V@1/test:', TEST_ACC[0])
                    print('V@5/test:', TEST_ACC[1])
                else:
                    print(self.output, " Not implemented")
            self.details = self.state['details']    
            self.best_acc = best_acc
            self.start_epoch = start_epoch + 1
            self.model.to(device)                    
        else:
            self.best_acc = -1.
            self.details = []   
            self.start_epoch = 0
    
    def eql_loss(self, pred_class_logits, gt):

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(1,self.num_classes + 1)
            target[:, gt_classes] = 1
            return target[:, :self.num_classes]

        target = expand_label(pred_class_logits, gt)

        beta = torch.Tensor([np.random.choice([0,1],p=[1-self.gamma, self.gamma])])
        beta = beta.expand(1, self.num_classes).to(device)
        
        weight = pred_class_logits.new_zeros(self.num_classes)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.num_classes)
        
        eql_w = 1 - beta * weight * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(pred_class_logits, target,
                                                      reduction='none')

        return torch.sum(cls_loss * eql_w) 

    
    def test(self):
        running_loss = 0.0
        correct_action = 0
        correct_verb = 0
        correct_noun = 0
        count = 0
        iterations = 0
        preds = []
        targets = []
        verb_preds = []
        noun_preds = []
        action_top1 = 0.
        action_top5 = 0.
        verb_targets = []
        noun_targets = []
        self.model.eval()
        for i, data in enumerate(self.testloader, 0): 
            #print(len(data))
            
        
            if self.dataset == 'ek55' or self.dataset == 'ek100' or self.dataset == 'egtea':
                seq_segs, action, verb, noun = data 
                feat_seq = []
                for seq in seq_segs: 
                    # print(seq)
                    feat_seq.append(seq)
                feat_seq = torch.stack(feat_seq)
                # feat_seq = (feat_seq - mean_values)/(std_values+1e-9)
                feat_seq = feat_seq.float().squeeze(0).to(device)
                # print(feat_seq.shape)
                if feat_seq.shape[1] == 0:
                    continue
                # print(action_label_tensor.is_cuda)
                out_next, out_cur, kld_obs_goal, kld_next_goal, kld_goal_diff = self.model(feat_seq)
                # print(out_next.shape)
                
                if self.output == 'verb':
                    preds.append(out_next.detach().cpu())
                    targets.append(verb[1])
                elif self.output == 'noun':
                    preds.append(out_next.detach().cpu())
                    targets.append(noun[1])
                elif self.output == 'action':
                    verb_scores = out_next[0].detach().cpu()
                    noun_scores = out_next[1].detach().cpu()
                    top5actions = get_top5actions(verb_scores.numpy(), noun_scores.numpy())
                    verb_preds.append(verb_scores)
                    noun_preds.append(noun_scores)
                    action_top1 += (action[1].item() == top5actions[0])
                    action_top5 += (action[1].item() in top5actions)
                    verb_targets.append(verb[1])
                    noun_targets.append(noun[1])
                elif self.output == 'act':
                    preds.append(out_next.detach().cpu())
                    targets.append(action[1])
                else:
                    print(self.output, "Not implemented")
            # print(next_actions.shape)

                   
            #loss = loss/len(obs_label_seq)
            print("\rIteration: {}/{}".format(i+1, len(self.testloader)), end="")
            #sys.stdout.flush()
            count += 1
            iterations += 1
                
            #print(count)   
        
        # plot_confusion(preds, targets)
        TEST_ACC = [] 
        if self.output == 'action':
            verb_preds = torch.stack(verb_preds)
            verb_targets = torch.stack(verb_targets)
            noun_preds = torch.stack(noun_preds)
            noun_targets = torch.stack(noun_targets)
            preds = [verb_preds, noun_preds]
            targets = [verb_targets, noun_targets]
            TEST_ACC.append(topk_accuracy(preds[0], targets[0], k=1).item()*100)
            TEST_ACC.append(topk_accuracy(preds[0], targets[0], k=5).item()*100) 
            TEST_ACC.append(topk_accuracy(preds[1], targets[1], k=1).item()*100)
            TEST_ACC.append(topk_accuracy(preds[1], targets[1], k=5).item()*100) 
            TEST_ACC.append(action_top1/count*100)
            TEST_ACC.append(action_top5/count*100) 
        else:
            preds = torch.stack(preds)
            targets = torch.stack(targets)

            TEST_ACC.append(topk_accuracy(preds, targets, k=1).item()*100)
            TEST_ACC.append(topk_accuracy(preds, targets, k=5).item()*100) 
        return TEST_ACC
        
    def train(self):        
        for epoch in range(self.start_epoch,self.nepoch):  
            start_time = time.time()        
            running_loss = 0.0
            correct_action = 0
            correct_verb = 0
            correct_noun = 0
            count = 0
            total_loss = 0
            self.optimizer.zero_grad()   
            loss = 0.
            iterations = 0
            count_unequal = 0
            self.model.train()
            for i, data in enumerate(self.trainloader, 0):        
                #print(len(data))
                     
                if self.dataset == 'ek55' or self.dataset == 'ek100' or self.dataset == 'egtea':
                    seq_segs, action, verb, noun = data 
                    feat_seq = []
                    for seq in seq_segs: 
                        #print(tsn_seq.shape)
                        feat_seq.append(seq)
                    feat_seq = torch.stack(feat_seq)
                    # feat_seq = (feat_seq - mean_values)/(std_values+1e-9)
                    feat_seq = feat_seq.float().squeeze(0).to(device)
                    # print(feat_seq.shape)
                    if feat_seq.shape[1] == 0:
                        continue
                    # # print(verb)
                    if self.output == "verb":
                        next_action = torch.LongTensor([ int(verb[1]) ]).to(device)
                        cur_action = torch.LongTensor([ int(verb[0]) ]).to(device)
                    elif self.output == 'noun':
                        next_action = torch.LongTensor([ int(noun[1]) ]).to(device)
                        cur_action = torch.LongTensor([ int(noun[0]) ]).to(device)
                    elif self.output == 'action':
                        next_verb = torch.LongTensor([ int(verb[1])]).to(device)
                        next_noun = torch.LongTensor([ int(noun[1])]).to(device)
                        next_action = torch.LongTensor([ int(action[1])]).to(device)
                        cur_verb = torch.LongTensor([ int(verb[0]) ]).to(device)
                        cur_noun = torch.LongTensor([ int(noun[0]) ]).to(device)
                    elif self.output == 'act':
                        next_action = torch.LongTensor([ int(action[1]) ]).to(device)
                        cur_action = torch.LongTensor([ int(action[0]) ]).to(device)
                    else:
                        print(self.output, "Not implemented")
                        
                    out_next, out_cur, kld_obs_goal, kld_next_goal, kld_goal_diff = self.model(feat_seq)
                # print(next_actions.shape)

                # print(action_label_tensor.is_cuda)
               
                # print('out_next elem:', len(out_next))
                if self.output == 'action':
                    # print(out_cur.shape)
                    pred_next = []
                    for out in out_next:
                        # print('out:', out.shape)
                        pred_next.append(torch.argmax(out,1))
                    
                    # Next action loss
                    # print('next_action: ', next_action.shape)
                    next_verb_loss = self.celoss(out_next[0], next_verb)
                    next_noun_loss = self.celoss(out_next[1], next_noun)
                    loss = 0.5*(next_verb_loss + next_noun_loss)
                    # Current action loss
                    cur_verb_loss = self.celoss(out_cur[0], cur_verb)
                    cur_noun_loss = self.celoss(out_cur[1], cur_noun)
                    loss += 0.5*(cur_verb_loss + cur_noun_loss)
                    # KL-Divergence Loss for observed latent goal
                    loss += kld_obs_goal
                    # KL-Divergence Loss for next latent goal
                    loss += kld_next_goal
                    # KL-Divergence Loss between latent_goals
                    loss += kld_goal_diff
                    with torch.no_grad():       
                        # self.writer.add_scalar('Next Verb Loss/train', self.celoss(out_next[0], next_verb).item(), epoch*self.trainset_size + i)
                        # self.writer.add_scalar('Next Noun Loss/train', self.celoss(out_next[1], next_noun).item(), epoch*self.trainset_size + i)
                        # self.writer.add_scalar('Cur Verb Loss/train', self.celoss(out_cur[0], cur_verb).item(), epoch*self.trainset_size + i)
                        # self.writer.add_scalar('Cur Noun Loss/train', self.celoss(out_cur[1], cur_noun).item(), epoch*self.trainset_size + i)
                        # self.writer.add_scalar('KLD Obs Lat Goal/train', kld_obs_goal.item(), epoch*self.trainset_size + i)
                        # self.writer.add_scalar('KLD Next Lat Goal/train', kld_next_goal.item(), epoch*self.trainset_size + i)
                        # self.writer.add_scalar('Sym KLD Goal Diff/train', kld_goal_diff.item(), epoch*self.trainset_size + i)
                        # self.writer.add_scalar('Total loss/train', loss.item(), epoch*self.trainset_size + i)
                        # print('pred_next:', pred_next)
                        correct_verb = correct_verb + torch.sum(pred_next[0] == next_verb).item()
                        correct_noun = correct_noun + torch.sum(pred_next[1] == next_noun).item()
                        pred_next_action = convert_to_action(pred_next[0].item(), pred_next[1].item())
                        if pred_next_action is not None:
                            pred_next_action = torch.LongTensor([pred_next_action]).to(device)
                            correct_action = correct_action + torch.sum(pred_next_action == next_action).item()
                        
                else:
                    # print(out_cur.shape)
                    pred_next = torch.argmax(out_next,1)
                    pred_cur = torch.argmax(out_cur,1)
                    
                    # print('out_next:', out_next.shape)    
                    # print('out_cur:', out_cur.shape)    
                    # print('next_action:', next_action)
                    # Next action loss
                    if 'na' in self.losses:
                        if next_action.item() != -1:
                            next_act_loss = self.celoss(out_next, next_action)
                            loss = next_act_loss
                            # self.writer.add_scalar('Pred Loss/train', next_act_loss.item(), epoch*self.trainset_size + i)
                    if 'eqna' in self.losses:
                        if next_action.item() != -1:
                            next_act_loss = self.eql_loss(out_next, next_action)
                            loss = next_act_loss
                            # self.writer.add_scalar('Pred Loss/train', next_act_loss.item(), epoch*self.trainset_size + i)
                    # print(loss)
                    # KL-Divergence Loss for observed latent goal
                    if 'og' in self.losses:
                        loss += kld_obs_goal
                        # self.writer.add_scalar('KLD Obs Lat Goal/train', kld_obs_goal.item(), epoch*self.trainset_size + i)
                    if 'ng' in self.losses:
                        # KL-Divergence Loss for next latent goal
                        loss += kld_next_goal
                        # print(kld_next_goal)
                        # self.writer.add_scalar('KLD Next Lat Goal/train', kld_next_goal.item(), epoch*self.trainset_size + i)
                        
                    if 'oa' in self.losses:
                        # Current action loss
                        if cur_action.item() != -1:
                            cur_act_loss = self.celoss(out_cur, cur_action)
                            loss += cur_act_loss
                            # self.writer.add_scalar('Cur Loss/train', cur_act_loss.item(), epoch*self.trainset_size + i)
                    if 'eqoa' in self.losses:
                        # Current action loss
                        if cur_action.item() != -1:
                            cur_act_loss = self.eql_loss(out_cur, cur_action)
                            loss += cur_act_loss
                            # self.writer.add_scalar('Cur Loss/train', cur_act_loss.item(), epoch*self.trainset_size + i)
                    if 'gc' in self.losses:
                        # KL-Divergence Loss between latent_goals  - Goal consistency
                        loss += kld_goal_diff
                        # self.writer.add_scalar('Sym KLD Goal Diff/train', kld_goal_diff.item(), epoch*self.trainset_size + i)
                        
                    with torch.no_grad():       
                            
                        # self.writer.add_scalar('Total loss/train', loss.item(), epoch*self.trainset_size + i)
                        
                        correct_action = correct_action + torch.sum(pred_next == next_action).item()

                loss.backward(retain_graph=True)
                
                 
                if i % (self.batch_size-1) == 0 and i>1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()    
                    if self.scheduler is not None:
                        self.scheduler.step()
                        print("\rIteration: {}/{}, LR: {}, Loss: {}.".format(i+1, len(self.trainloader), self.scheduler.get_last_lr(), loss), end="")
                    else:
                        print("\rIteration: {}/{}, LR: {}, Loss: {}.".format(i+1, len(self.trainloader), self.optimizer.param_groups[0]['lr'], loss), end="")
                    running_loss = running_loss + loss.item()
                    loss = 0.
                    count += self.batch_size
                    iterations += 1
            if self.scheduler is not None:        
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.steps)
            # print('count_unequal',count_unequal)
            if self.output == 'action':
                TRAIN_LOSS = running_loss/iterations
                TRAIN_ACC = []
                TRAIN_ACC.append(correct_verb/count*100)
                TRAIN_ACC.append(correct_noun/count*100)
                TRAIN_ACC.append(correct_action/count*100)
                # self.writer.add_scalar('V/train', TRAIN_ACC[0], epoch+1)
                # self.writer.add_scalar('N/train', TRAIN_ACC[1], epoch+1)
                # self.writer.add_scalar('A/train', TRAIN_ACC[2], epoch+1)
                
                TEST_ACC = self.test()
                                
                self.details.append((TRAIN_LOSS, TRAIN_ACC, 0., TEST_ACC))
                
                # self.writer.add_scalar('V@1/test', TEST_ACC[0], epoch+1)
                # self.writer.add_scalar('V@5/test', TEST_ACC[1], epoch+1)
                # self.writer.add_scalar('N@1/test', TEST_ACC[2], epoch+1)
                # self.writer.add_scalar('N@5/test', TEST_ACC[3], epoch+1)
                # self.writer.add_scalar('A@1/test', TEST_ACC[4], epoch+1)
                # self.writer.add_scalar('A@5/test', TEST_ACC[5], epoch+1)
                
                if TEST_ACC[4] > self.best_acc:                
                    self.state = {
                        'model': self.model.state_dict(),
                        'acc': TEST_ACC[4],
                        'epoch': epoch,
                        'details':self.details,            
                    }        
                    torch.save(self.state, self.chkpath)
                    self.best_acc = TEST_ACC[0]
                else:
                    self.state['epoch'] = epoch
                    torch.save(self.state, self.chkpath)
                elapsed_time = time.time() - start_time
                print('[{}] [{:.1f}] [Loss {:.3f}] [A@1 {:.2f}]'.format(epoch, elapsed_time,
                        TRAIN_LOSS, TRAIN_ACC[2]),end=" ")
                print('[V@1 {:.2f}], [V@5 {:.2f}], [N@1 {:.2f}], [N@5 {:.2f}], [A@1 {:.2f}], [A@5 {:.2f}]'.format(\
                       TEST_ACC[0], TEST_ACC[1], TEST_ACC[2], TEST_ACC[3], TEST_ACC[4], TEST_ACC[5]))
            else:
                TRAIN_LOSS = running_loss/iterations
                TRAIN_ACC = correct_action/count*100
                TEST_ACC = self.test()
                self.details.append((TRAIN_LOSS,TRAIN_ACC,0.,TEST_ACC))
                # self.writer.add_scalar('Acc/train', TRAIN_ACC, epoch+1)
                # self.writer.add_scalar('Acc/test', TEST_ACC[0], epoch+1)
                # self.writer.add_scalar('Acc@5/test', TEST_ACC[1], epoch+1)
            
                if TEST_ACC[0] > self.best_acc:                
                    self.state = {
                        'model': self.model.state_dict(),
                        'acc': TEST_ACC[0],
                        'epoch': epoch,
                        'details':self.details,            
                    }        
                    torch.save(self.state, self.chkpath)
                    self.best_acc = TEST_ACC[0]
                else:
                    self.state['epoch'] = epoch
                    torch.save(self.state, self.chkpath)
                elapsed_time = time.time() - start_time
                print('[{}] [{:.1f}] [Loss {:.3f}] [Train Acc {:.2f}]'.format(epoch, elapsed_time,
                        TRAIN_LOSS, TRAIN_ACC),end=" ")
                print('[Test Acc @1 {:.2f}], [Test Acc @5 {:.2f}]'.format(TEST_ACC[0], TEST_ACC[1]))
            
