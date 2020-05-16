import numpy as np 
import os 
import scipy.io
from sklearn.metrics import average_precision_score, precision_recall_curve

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
import torchvision.models as models
from torchvision import transforms

from emotic_dataset import Emotic_PreDataset


def test_scikit_ap(cat_preds, cat_labels, ind2cat):
  ''' Calculate average precision per emotion category using sklearn library.
  :param cat_preds: Categorical emotion predictions. 
  :param cat_labels: Categorical emotion labels. 
  :param ind2cat: Dictionary converting integer index to categorical emotion.
  :return: Numpy array containing average precision per emotion category.
  '''
  ap = np.zeros(26, dtype=np.float32)
  for i in range(26):
    ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
    print ('Category %16s %.5f' %(ind2cat[i], ap[i]))
  print ('Mean AP %.5f' %(ap.mean()))
  return ap 


def test_vad(cont_preds, cont_labels, ind2vad):
  ''' Calcaulate VAD (valence, arousal, dominance) errors. 
  :param cont_preds: Continuous emotion predictions. 
  :param cont_labels: Continuous emotion labels. 
  :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
  :return: Numpy array containing mean absolute error per continuous emotion dimension. 
  '''
  vad = np.zeros(3, dtype=np.float32)
  for i in range(3):
    vad[i] = np.mean(np.abs(cont_preds[i, :] - cont_labels[i, :]))
    print ('Continuous %10s %.5f' %(ind2vad[i], vad[i]))
  print ('Mean VAD Error %.5f' %(vad.mean()))
  return vad


def get_thresholds(cat_preds, cat_labels):
  ''' Calculate thresholds where precision is equal to recall. These thresholds are then later for inference.
  :param cat_preds: Categorical emotion predictions. 
  :param cat_labels: Categorical emotion labels. 
  :return: Numpy array containing thresholds per emotion category where precision is equal to recall.
  '''
  thresholds = np.zeros(26, dtype=np.float32)
  for i in range(26):
    p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
    for k in range(len(p)):
      if p[k] == r[k]:
        thresholds[i] = t[k]
        break
  return thresholds


def test_data(models, device, data_loader, ind2cat, ind2vad, num_images, result_dir='./', test_type='val'):
    ''' Test models on data 
    :param models: List containing model_context, model_body and emotic_model (fusion model) in that order.
    :param device: Torch device. Used to send tensors to GPU if available. 
    :param data_loader: Dataloader iterating over dataset. 
    :param ind2cat: Dictionary converting integer index to categorical emotion.
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance)
    :param num_images: Number of images in the dataset. 
    :param result_dir: Directory path to save results (predictions mat object and thresholds npy object).
    :param test_type: Test type variable. Variable used in the name of thresholds and predictio files.
    '''
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


def test_emotic(result_path, model_path, ind2cat, ind2vad, context_norm, body_norm, args):
    ''' Prepare test data and test models on the same.
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training. 
    :param ind2cat: Dictionary converting integer index to categorical emotion. 
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :param context_norm: List containing mean and std values for context images. 
    :param body_norm: List containing mean and std values for body images. 
    :param args: Runtime arguments.
    '''    
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
