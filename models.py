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

torch.manual_seed(14)
np.random.seed(14)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)


from collections import defaultdict
import pickle
import random
import pandas as pd

class VarAnt(nn.Module):
    def __init__(self, num_classes, args, feat_dim=None):
        super(VarAnt, self).__init__()
        self.feat_dim = feat_dim
                    
        self.h_dim = args.hidden_dim # hidden dimension for RNNs
        self.z_dim = args.latent_dim # latent dimension for goal
        self.num_act_cand = args.num_act_cand # number of action candidates used for nextiction
        self.n_layers = args.n_layers # number of layers in sampler RNN for generating next action latent goal
        self.num_goal_cand = args.num_goal_cand # number of samples drawn from observed goal distribution
        self.num_classes = num_classes # number of output classes 
        # # # Linear layer for observed feature x_t
        self.phi_x = nn.Sequential(nn.Linear(self.feat_dim, self.h_dim),
                                   nn.ReLU())
        # # # Linear layer for encoder - observed feature x_t and hidden state from t-1
        self.phi_enc = nn.Sequential(nn.Linear(self.h_dim + self.h_dim, self.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.h_dim, self.z_dim))
        # # #  MLP for prior - input is hidden state from t-1
        self.phi_prior = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim))
        # # #  Linear layer for latent goal z_t at time t
        self.phi_z = nn.Sequential(nn.Linear(self.z_dim, self.h_dim),
                                   nn.ReLU())
                                   
        # # # RNN to generate prior and encoder distributions for observed features
        self.rnn = nn.GRU(2*self.h_dim, self.h_dim, 1)
        
        # # # After getting the last latent goal change phi_z for sample
        self.phi_z_new = nn.Sequential(nn.Linear(self.z_dim, self.h_dim),
                                   nn.ReLU())
        # # # After getting the last layer change phi_prior 
        self.phi_prior_new = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim))
        # # # Layer normalization on rnn output
        # self.layernorm = nn.LayerNorm(self.h_dim)
        
        # # # Linear layer for Observed action based on latent goal and final hidden state 
        # # # OR
        # # # Linear layer for nexticted action candidate based on observed action and final hidden state 
        self.phi_get_act = nn.Sequential(nn.Linear(self.z_dim + self.h_dim, self.h_dim),
                                   nn.ReLU())

        # # #  RNN to generate nexticted visual feature for nexticted latent goal
        # self.sampler_rnn = nn.GRU(h_dim, h_dim, self.n_layers)
        
        # # # Linear layer for nexticted latent goal 
        # self.phi_prior_next = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU())
        
        # # # Linear layer to transform observed action to z_dim each
        self.phi_obs_act = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim))
                                       
        # # # Linear layer to transform next action to z_dim
        self.phi_next_act = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim))
        
        # # # Linear layer for next latent goal - inputs are next and observed action
        self.phi_enc_next = nn.Sequential(nn.Linear(2*self.z_dim, self.z_dim), nn.ReLU())
        
        # # # For generating std of distributions
        self.softplus = nn.Softplus()
    
        self.softmax = nn.Softmax(dim=-1)
        
        self.cur_act_classifier = nn.Linear(self.h_dim, self.num_classes)
        self.next_act_classifier = nn.Linear(self.h_dim, self.num_classes)

                
    def init_hidden(self, x):
        return torch.randn(self.n_layers, x.size(1), self.h_dim).to(device)    
    
    def reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.randn(std.shape, requires_grad=True).to(device)
        return eps.mul(std).add_(mean)

    def kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return .5 * torch.sum(kld_element)
    
    def sym_kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element_1 = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
                       
        kld_element_2 = (2 * torch.log(std_1) - 2 * torch.log(std_2) +
                       (std_2.pow(2) + (mean_2 - mean_1).pow(2)) /
                       std_1.pow(2) - 1)
        return .5 * (torch.sum(kld_element_1) + torch.sum(kld_element_2))

    def forward(self, feat_seq, batch_size=None):
      
        kld_lat_goal = 0 
        if len(feat_seq.shape) == 2:
            feat_seq = feat_seq.unsqueeze(0)
        h = self.init_hidden(feat_seq)
        # # # conditional VRNN for generating latent goal based on observed features
       
        for t in range(feat_seq.shape[0]):
            # print('feat dim:',self.feat_dim)
            x_t = feat_seq[t,:]
            # print('x_t:',x_t.shape)
            # # # Encoder distribution that combines observed feature at time t and hidden state from t-1
            try:
                enc_t = self.phi_enc(torch.cat([h[-1], self.phi_x(x_t)], -1))
            except:
                print('feat_seq:', feat_seq.shape)
                print('x_t:', self.phi_x(x_t).shape)
                print('h[-1]:', h[-1].shape)
            enc_mean_t = enc_t
            enc_std_t = self.softplus(enc_t)
            
            # # # Prior distribution based on hidden state from t-1
            prior_t = self.phi_prior(h[-1])
            prior_mean_t = prior_t
            prior_std_t = self.softplus(prior_t)

            # # # Comparing KLDiv between encoder and prior distributions
            kld_lat_goal += self.kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            # # # Latent goal at time t
            z_t = self.reparameterized_sample(enc_mean_t, enc_std_t)

            _, h = self.rnn(torch.cat([self.phi_x(x_t).unsqueeze(0), self.phi_z(z_t).unsqueeze(0)], -1), h)
            
            # # #  applying layernorm on hidden state
            # h = self.layernorm(h)
        # # # Distribution for latent goal for observed sequence
        obs_lat_goal_mean = self.phi_prior(h[-1])
        obs_lat_goal_std = self.softplus(self.phi_prior(h[-1]))
        
        lat_goal_dis = 1e9
        next_act_final = 0.
        obs_act_final = 0.
        kld_next_lat_goal = 0.
        for _ in range(self.num_goal_cand):
            # # # Sample latent goal for observed sequence
            obs_lat_goal = self.reparameterized_sample(obs_lat_goal_mean, obs_lat_goal_std)

            # # # Observed action based on latent goal and final hidden state 
            obs_act = self.phi_get_act(torch.cat([self.phi_z_new(obs_lat_goal).unsqueeze(0), self.phi_prior_new(h[-1]).unsqueeze(0)], -1))
            
            # # # Predicted(next) action using observed action and final hidden state
            next_act_dummy = self.phi_get_act(torch.cat([self.phi_prior_new(h[-1]).unsqueeze(0), obs_act], -1))
            next_act_mean = next_act_dummy
            next_act_std = self.softplus(next_act_dummy)
            
            
            for i in range(self.num_act_cand): 
                # # # Next action sample
                next_act = self.reparameterized_sample(next_act_mean, next_act_std)
                # print('next_act:', next_act.shape)
                # print('obs_act:', obs_act.shape)
                
                # CVAE for next action 
                # # # Prior distribution for next latent goal
                prior_next_lat_goal_mean = self.phi_next_act(next_act)
                prior_next_lat_goal_std = self.softplus(self.phi_next_act(next_act))
                # print('prior_next_lat_goal_std:', prior_next_lat_goal_std.shape)
                
                # # # Encoder distribution for nexticted latent goal conditioned on observed action
                enc_next_lat_goal = self.phi_enc_next(torch.cat([self.phi_next_act(next_act), self.phi_obs_act(obs_act)], -1))
                enc_next_lat_goal_mean = enc_next_lat_goal
                enc_next_lat_goal_std = self.softplus(enc_next_lat_goal)
                # print('enc_next_lat_goal_std:', enc_next_lat_goal_std.shape)
                
                # # # Symmetric KLDiv between latent goal distribution for observed action and nexticted action
                new_lat_dis = self.sym_kld_gauss( obs_lat_goal_mean, obs_lat_goal_std, \
                                                  enc_next_lat_goal_mean, enc_next_lat_goal_std)
                
                if i == 0:
                    lat_goal_dis = new_lat_dis
                    next_act_final = next_act
                    obs_act_final = obs_act
                    kld_next_lat_goal = self.kld_gauss( enc_next_lat_goal_mean, enc_next_lat_goal_std,\
                                               prior_next_lat_goal_mean, prior_next_lat_goal_std)
                # # # The action candidate with minimal Symmetric KLD is chosen
                if new_lat_dis < lat_goal_dis:
                    
                    lat_goal_dis = new_lat_dis
                    # # # Comparing KLDiv between encoder and prior distributions of nexticted latent goal
                    kld_next_lat_goal = self.kld_gauss( enc_next_lat_goal_mean, enc_next_lat_goal_std,\
                                               prior_next_lat_goal_mean, prior_next_lat_goal_std)
                    # # # Predicted action
                    next_act_final = next_act
                    obs_act_final = obs_act
        # # # Embedding both next and current action to number of action classes
        try:
            out_next = self.next_act_classifier(next_act_final)    
        except:
            print('next_act_final:', next_act_final.shape)   
        out_cur = self.cur_act_classifier(obs_act_final)
        
        return out_next.squeeze(0), out_cur.squeeze(0), kld_lat_goal, kld_next_lat_goal, lat_goal_dis
