import torch
import torch.nn as nn

class DiscreteLoss(nn.Module):
  ''' Class to measure loss between categorical emotion predictions and labels.'''
  def __init__(self, weight_type='mean', device=torch.device('cpu')):
    super(DiscreteLoss, self).__init__()
    self.weight_type = weight_type
    self.device = device
    if self.weight_type == 'mean':
      self.weights = torch.ones((1,26))/26.0
      self.weights = self.weights.to(self.device)
    elif self.weight_type == 'static':
      self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
         0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
         0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537]).unsqueeze(0)
      self.weights = self.weights.to(self.device)
    
  def forward(self, pred, target):
    if self.weight_type == 'dynamic':
      self.weights = self.prepare_dynamic_weights(target)
      self.weights = self.weights.to(self.device)
    loss = (((pred - target)**2) * self.weights)
    return loss.sum() 

  def prepare_dynamic_weights(self, target):
    target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
    weights = torch.zeros((1,26))
    weights[target_stats != 0 ] = 1.0/torch.log(target_stats[target_stats != 0].data + 1.2)
    weights[target_stats == 0] = 0.0001
    return weights


class ContinuousLoss_L2(nn.Module):
  ''' Class to measure loss between continuous emotion dimension predictions and labels. Using l2 loss as base. '''
  def __init__(self, margin=1):
    super(ContinuousLoss_L2, self).__init__()
    self.margin = margin
  
  def forward(self, pred, target):
    labs = torch.abs(pred - target)
    loss = labs ** 2 
    loss[ (labs < self.margin) ] = 0.0
    return loss.sum()


class ContinuousLoss_SL1(nn.Module):
  ''' Class to measure loss between continuous emotion dimension predictions and labels. Using smooth l1 loss as base. '''
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

  disc_loss = DiscreteLoss('dynamic', torch.device("cuda:0"))
  loss = disc_loss(pred, target)
  print ('discrete loss class', loss, loss.shape, loss.dtype, loss.requires_grad)  # loss = 37.1217

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
  print ('continuous SL1 loss class', loss, loss.shape, loss.dtype, loss.requires_grad) # loss = 4.8750

  cont_loss_L2 = ContinuousLoss_L2()
  loss = cont_loss_L2(pred*10, target * 10)
  print ('continuous L2 loss class', loss, loss.shape, loss.dtype, loss.requires_grad) # loss = 12.0
