import torch
import numpy as np 

''' Loss for categorical labels (Emotion classes) '''
def get_discreteloss(pred, target, device='cpu', weight_type='mean'):
  if weight_type == 'mean':
    weights = torch.ones((1,26))/26.0
  elif weight_type == 'dynamic':
    target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
    weights = torch.zeros((1,26))
    weights[target_stats != 0 ] = 1.0/torch.log(target_stats[target_stats != 0].data + 1.2)
    weights[target_stats == 0] = 0.0001

  if device == 'gpu':
    weights = weights.cuda()
  loss = (((pred - target)**2) * weights)
  return loss.sum() 

''' L2 loss for continuous labels (Valence, Arousal, Dominance) '''
def get_continuousloss_L2(pred, target, margin=1):
  labs = torch.abs(pred - target) 
  loss = labs ** 2
  loss[labs < margin] = 0.0
  return loss.sum()
  
''' Smooth L1 loss for continuous labels (Valence, Arousal, Dominance) '''
def get_continuousloss_SL1(pred, target, margin=1):
  labs = torch.abs(pred - target) 
  loss = 0.5 * (labs ** 2)
  loss[labs > margin] = labs[labs > margin] - 0.5
  return loss.sum()
  
