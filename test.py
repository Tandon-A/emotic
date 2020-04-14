import numpy as np 
import os 
import scipy.io
from sklearn.metrics import average_precision_score
import torch

''' Calculate avergae precision per category using sklearn library '''
def test_scikit_ap(cat_preds, cat_labels, ind2cat):
  ap = np.zeros(26, dtype=np.float32)
  for i in range(26):
    ap[i] = average_precision_score(cat_labels[:, i], cat_preds[:, i])
    print ('Category %16s %.5f' %(ind2cat[i], ap[i]))
  print ('Mean AP %.5f' %(ap.mean()))



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
    
    ''' Mat files used for emotic testing (matlab script)'''
    if save_results == True:
        scipy.io.savemat(os.path.join(result_dir, 'cat_preds.mat'), mdict={'cat_preds':cat_preds})
        scipy.io.savemat(os.path.join(result_dir, 'cat_labels.mat'), mdict={'cat_labels':cat_labels})
        print ('saved mat files')

    test_scikit_ap(cat_preds, cat_labels, ind2cat)

