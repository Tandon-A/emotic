import torch 
import torch.nn as nn 

''' Emotic Model'''
class Emotic18(nn.Module):
  def __init__(self):
    super(Emotic18,self).__init__()
    self.fc1 = nn.Linear(1024,256)
    self.bn1 = nn.BatchNorm1d(256)
    self.d1 = nn.Dropout(p=0.5)
    self.fc_cat = nn.Linear(256,26)
    self.fc_cont = nn.Linear(256,3)
    self.relu = nn.ReLU()

  def forward(self, x_context, x_body):
    context_features = x_context.view(-1,512)
    body_features = x_body.view(-1,512)
    fuse_features = torch.cat((context_features,body_features),1)
    fuse_out = self.fc1(fuse_features)
    fuse_out = self.bn1(fuse_out)
    fuse_out = self.relu(fuse_out)
    fuse_out = self.d1(fuse_out)    
    cat_out = self.fc_cat(fuse_out)
    cont_out = self.fc_cont(fuse_out)
    return cat_out, cont_out 
