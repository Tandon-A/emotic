import numpy as np 
import os 
import scipy.io
from sklearn.metrics import average_precision_score

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
    ap[i] = average_precision_score(cat_labels[:, i], cat_preds[:, i])
    print ('Category %16s %.5f' %(ind2cat[i], ap[i]))
  print ('Mean AP %.5f' %(ap.mean()))

''' Test models on data '''
def test_data(models, device, data_loader, ind2cat, num_images, save_results=False, result_dir='./'):
    model_context, model_body, emotic_model = models
    cat_preds = np.zeros((num_images, 26))
    cat_labels = np.zeros((num_images, 26))

    with torch.no_grad():
        model_context.to(device)
        model_body.to(device)
        emotic_model.to(device)
        model_context.eval()
        model_body.eval()
        emotic_model.eval()
        indx = 0
        print ('starting testing')
        for images_context, images_body, labels_cat, _ in iter(data_loader):
            images_context = images_context.to(device)
            images_body = images_body.to(device)

            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_cat, _ = emotic_model(pred_context, pred_body)

            cat_preds[ indx : (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
            cat_labels[ indx : (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
            indx = indx + pred_cat.shape[0]

    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    print ('completed testing')
    
    # Mat files used for emotic testing (matlab script)
    if save_results == True:
        scipy.io.savemat(os.path.join(result_dir, 'cat_preds.mat'), mdict={'cat_preds':cat_preds})
        scipy.io.savemat(os.path.join(result_dir, 'cat_labels.mat'), mdict={'cat_labels':cat_labels})
        print ('saved mat files')

    test_scikit_ap(cat_preds, cat_labels, ind2cat)

''' Prepare test data and test models on the same'''
def test_emotic(result_path, model_path, ind2cat, context_norm, body_norm, args):

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
    test_data([model_context, model_body, emotic_model], device, test_loader, ind2cat, test_dataset.__len__(), save_results=True, result_dir=result_path)
