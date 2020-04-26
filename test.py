import numpy as np 
import os 
import scipy.io
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
import torchvision.models as models
from torchvision import transforms

from emotic_dataset import Emotic_PreDataset

''' Calculate average precision per category using sklearn library. '''
def test_scikit_ap(cat_preds, cat_labels, ind2cat):
  ap = np.zeros(26, dtype=np.float32)
  for i in range(26):
    ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
    print ('Category %16s %.5f' %(ind2cat[i], ap[i]))
  print ('Mean AP %.5f' %(ap.mean()))
  return ap 

''' Calcaulate VAD (valence, arousal, dominance) errors. '''
def test_vad(cont_preds, cont_labels, ind2vad):
  vad = np.zeros(3, dtype=np.float32)
  for i in range(3):
    vad[i] = np.mean(np.abs(cont_preds[i, :] - cont_labels[i, :]))
    print ('Continuous %10s %.5f' %(ind2vad[i], vad[i]))
  print ('Mean VAD Error %.5f' %(vad.mean()))
  return vad

''' Calculate thresholds where precision = recall. Used for inference. '''
def get_thresholds(cat_preds, cat_labels):
  thresholds = np.zeros(26, dtype=np.float32)
  for i in range(26):
    p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
    for k in range(len(p)):
      if p[k] == r[k]:
        thresholds[i] = t[k]
        break
  return thresholds

''' Test models on data '''
def test_data(models, device, data_loader, ind2cat, ind2vad, num_images, result_dir='./', test_type='val'):
    model_context, model_body, emotic_model = models
    cat_preds = np.zeros((num_images, 26))
    cat_labels = np.zeros((num_images, 26))
    cont_preds = np.zeros((num_images, 3))
    cont_labels = np.zeros((num_images, 3))

    with torch.no_grad():
        model_context.to(device)
        model_body.to(device)
        emotic_model.to(device)
        model_context.eval()
        model_body.eval()
        emotic_model.eval()
        indx = 0
        print ('starting testing')
        for images_context, images_body, labels_cat, labels_cont in iter(data_loader):
            images_context = images_context.to(device)
            images_body = images_body.to(device)

            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_cat, pred_cont = emotic_model(pred_context, pred_body)

            cat_preds[ indx : (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
            cat_labels[ indx : (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
            cont_preds[ indx : (indx + pred_cont.shape[0]), :] = pred_cont.to("cpu").data.numpy() * 10
            cont_labels[ indx : (indx + labels_cont.shape[0]), :] = labels_cont.to("cpu").data.numpy() * 10
            indx = indx + pred_cat.shape[0]

    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    cont_preds = cont_preds.transpose()
    cont_labels = cont_labels.transpose()
    print ('completed testing')
    
    # Mat files used for emotic testing (matlab script)
    scipy.io.savemat(os.path.join(result_dir, '%s_cat_preds.mat' %(test_type)), mdict={'cat_preds':cat_preds})
    scipy.io.savemat(os.path.join(result_dir, '%s_cat_labels.mat' %(test_type)), mdict={'cat_labels':cat_labels})
    scipy.io.savemat(os.path.join(result_dir, '%s_cont_preds.mat' %(test_type)), mdict={'cont_preds':cont_preds})
    scipy.io.savemat(os.path.join(result_dir, '%s_cont_labels.mat' %(test_type)), mdict={'cont_labels':cont_labels})
    print ('saved mat files')

    test_scikit_ap(cat_preds, cat_labels, ind2cat)
    test_vad(cont_preds, cont_labels, ind2vad)
    thresholds = get_thresholds(cat_preds, cat_labels)
    np.save(os.path.join(result_dir, '%s_thresholds.npy' %(test_type)), thresholds)
    print ('saved thresholds')


''' Prepare test data and test models on the same'''
def test_emotic(result_path, model_path, ind2cat, ind2vad, context_norm, body_norm, args):

    # Prepare models 
    model_context = torch.load(os.path.join(model_path,'model_context1.pth'))
    model_body = torch.load(os.path.join(model_path,'model_body1.pth'))
    emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth'))
    print ('Succesfully loaded models')

    #Load data preprocessed npy files
    test_context = np.load(os.path.join(args.data_path, 'test_context_arr.npy'))
    test_body = np.load(os.path.join(args.data_path, 'test_body_arr.npy'))
    test_cat = np.load(os.path.join(args.data_path, 'test_cat_arr.npy'))
    test_cont = np.load(os.path.join(args.data_path, 'test_cont_arr.npy'))
    print ('test ', 'context ', test_context.shape, 'body', test_body.shape, 'cat ', test_cat.shape, 'cont', test_cont.shape)

    # Initialize Dataset and DataLoader 
    test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    test_dataset = Emotic_PreDataset(test_context, test_body, test_cat, test_cont, test_transform, context_norm, body_norm)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    print ('test loader ', len(test_loader))
    
    device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
    test_data([model_context, model_body, emotic_model], device, test_loader, ind2cat, ind2vad, len(test_dataset), result_dir=result_path, test_type='test')
