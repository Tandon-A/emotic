import torch
import torch.nn as nn

''' Loss for categorical labels (Emotion classes) '''
class DiscreteLoss(nn.Module):
  def __init__(self, weight_type='mean'):
    super(DiscreteLoss, self).__init__()
    self.weight_type = weight_type
    if self.weight_type == 'mean':
      self.weights = torch.ones((1,26))/26.0
    # elif self.weight_type == 'static':

  def forward(self, pred, target, device='cpu'):
    if self.weight_type == 'dynamic':
      self.weights = self.prepare_dynamic_weights(target)
    print ('weights class', self.weights.shape, self.weights.dtype, self.weights.requires_grad)
    if device == 'gpu':
      self.weights = self.weights.cuda()
    loss = (((pred - target)**2) * self.weights)
    return loss.sum() 

  def prepare_dynamic_weights(self, target):
    target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
    weights = torch.zeros((1,26))
    weights[target_stats != 0 ] = 1.0/torch.log(target_stats[target_stats != 0].data + 1.2)
    weights[target_stats == 0] = 0.0001
    return weights


''' L2 loss for continuous labels (Valence, Arousal, Dominance) '''
class ContinuousLoss_L2(nn.Module):
  def __init__(self, margin=1):
    super(ContinuousLoss_L2, self).__init__()
    self.margin = margin
  
  def forward(self, pred, target):
    labs = torch.abs(pred - target)
    loss = labs ** 2 
    loss[ (labs < self.margin) ] = 0.0
    return loss.sum()
 

''' Smooth L1 loss for continuous labels (Valence, Arousal, Dominance) '''
class ContinuousLoss_SL1(nn.Module):
  def __init__(self, margin=1):
    super(ContinuousLoss_SL1, self).__init__()
    self.margin = margin
  
  def forward(self, pred, target):
    labs = torch.abs(pred - target)
    loss = 0.5 * (labs ** 2)
    loss[ (labs > self.margin) ] = labs[ (labs > self.margin) ] - 0.5
    return loss.sum()


if __name__ == '__main__':
  # Discrete Loss function test 
  target = torch.zeros((2,26))
  target[0, 0:13] = 1
  target[1, 13:] = 2
  target[:, 13] = 0
  
  pred = torch.ones((2,26)) * 1
  target = target.cuda()
  pred = pred.cuda()
  pred.requires_grad = True
  target.requires_grad = False

  disc_loss = DiscreteLoss('dynamic')
  loss = disc_loss(pred, target, device = 'gpu')
  print (' discrete loss class', loss, loss.shape, loss.dtype, loss.requires_grad)  # loss = 37.1217

  #Continuous Loss function test
  target = torch.ones((2,3))
  target[0, :] = 0.9
  target[1, :] = 0.2
  target = target.cuda()
  pred = torch.ones((2,3))
  pred[0, :] = 0.7
  pred[1, :] = 0.25
  pred = pred.cuda()
  pred.requires_grad = True
  target.requires_grad = False

  cont_loss_SL1 = ContinuousLoss_SL1()
  loss = cont_loss_SL1(pred*10, target * 10)
  print (' continuous SL1 loss class', loss, loss.shape, loss.dtype, loss.requires_grad) # loss = 4.8750

  cont_loss_L2 = ContinuousLoss_L2()
  loss = cont_loss_L2(pred*10, target * 10)
  print (' continuous L2 loss class', loss, loss.shape, loss.dtype, loss.requires_grad) # loss = 12.0
  
