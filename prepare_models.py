import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F
import os


def prep_models(context_model='resnet18', body_model='resnet18', model_dir='./'):
  model_name = '%s_places365.pth.tar' % context_model
  model_file = os.path.join(model_dir, model_name)
  if not os.path.exists(model_file):
    download_command = 'wget ' + 'http://places2.csail.mit.edu/models_places365/' + model_name +' -O ' + model_file
    os.system(download_command)

  save_file = os.path.join(model_dir,'%s_places365_py36.pth.tar' % context_model)
  from functools import partial
  import pickle
  pickle.load = partial(pickle.load, encoding="latin1")
  pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
  model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
  torch.save(model, save_file)
  print('converting %s -> %s'%(model_file, save_file))

  # create the network architecture
  model = models.__dict__[context_model](num_classes=365)

  checkpoint = torch.load(save_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!
  state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()} # the data parallel layer will add 'module' before each layer name
  model.load_state_dict(state_dict)
  model.eval()
  model.cpu()
  torch.save(model, os.path.join(model_dir, 'context_model' + '.pth'))
  
  print ('completed preparing context model')


prep_models(model_dir='./')


