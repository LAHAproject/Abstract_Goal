""" Set of utilities """
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score
import pandas as pd

def topk_accuracy(inp, targ, k=5, axis=-1):
    "Computes the Top-k accuracy (`targ` is in the top `k` predictions of `inp`)"
    inp = inp.topk(k=k, dim=axis)[1]
    targ = targ.unsqueeze(dim=axis).expand_as(inp)
    return (inp == targ).sum(dim=-1).float().mean()

def plot_confusion(inp, targ):
    _, preds = torch.max(inp, -1)
    preds = preds.numpy()
    targ = targ.numpy()
    print(targ.shape)
    print('\n')
    print(confusion_matrix(targ, preds))
    print('\n')
    print(recall_score(targ, preds, average='macro'))

def load_pretrained_model(model, ckpt):
    state = torch.load(ckpt)
    model.load_state_dict(state['model'], strict=False)
    for param in model.parameters():
        param.requires_grad = False
    return model

def convert_to_action(verb, noun):
    action_mapping = pd.read_csv('../EPIC_KITCHENS_2020/actions.csv')
    try:
        action = action_mapping.loc[(action_mapping['verb'] == verb) & (action_mapping['noun'] == noun)]['id'].item()
    except:
        action = None
    return action

def get_top5actions(action_scores):
    val, idx = torch.topk(action_scores.flatten(), 100)
    topactions = (np.array(np.unravel_index(idx.numpy(), action_scores.shape)).T)
    
    action_mapping = pd.read_csv('../EPIC_KITCHENS_2020/actions.csv')
    top5actions = []
    for verb, noun in topactions:
        if len(top5actions) == 5:
            break
        try:
            action = action_mapping.loc[(action_mapping['verb'] == verb) & (action_mapping['noun'] == noun)]['id'].item()
            top5actions.append(action)
        except:
            continue
    return top5actions

