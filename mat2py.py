import argparse 
import csv 
import cv2
import numpy as np 
import os 
from scipy.io import loadmat 


class emotic_train:
    def __init__(self, filename, folder, image_size, person):
        self.filename = filename
        self.folder = folder
        self.im_size = []
        self.bbox = []
        self.cat = []
        self.cont = []
        self.gender = person[3][0]
        self.age = person[4][0]
        self.cat_annotators = 0
        self.cont_annotators = 0
        self.set_imsize(image_size)
        self.set_bbox(person[0])
        self.set_cat(person[1])
        self.set_cont(person[2])
        self.check_cont()

    def set_imsize(self, image_size):
        image_size = np.array(image_size).flatten().tolist()[0]
        row = np.array(image_size[0]).flatten().tolist()[0]
        col = np.array(image_size[1]).flatten().tolist()[0]
        self.im_size.append(row)
        self.im_size.append(col)

    def validate_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        x1 = min(self.im_size[0], max(0, x1))
        x2 = min(self.im_size[0], max(0, x2))
        y1 = min(self.im_size[1], max(0, y1))
        y2 = min(self.im_size[1], max(0, y2))
        return [int(x1), int(y1), int(x2), int(y2)]

    def set_bbox(self, person_bbox):
        self.bbox = self.validate_bbox(np.array(person_bbox).flatten().tolist())

    def set_cat(self, person_cat):
        cat = np.array(person_cat).flatten().tolist()
        cat = np.array(cat[0]).flatten().tolist()
        self.cat = [np.array(c).flatten().tolist()[0] for c in cat]
        self.cat_annotators = 1

    def set_cont(self, person_cont):
        cont = np.array(person_cont).flatten().tolist()[0]
        self.cont = [np.array(c).flatten().tolist()[0] for c in cont]
        self.cont_annotators = 1

    def check_cont(self):
        for c in self.cont:
            if np.isnan(c):
                self.cont_annotators = 0
                break

class emotic_test:
    def __init__(self, filename, folder, image_size, person):
        self.filename = filename
        self.folder = folder
        self.im_size = []
        self.bbox = []
        self.cat = []
        self.cat_annotators = 0
        self.comb_cat = []
        self.cont_annotators = 0
        self.cont = []
        self.comb_cont = []
        self.gender = person[5][0]
        self.age = person[6][0]

        self.set_imsize(image_size)
        self.set_bbox(person[0])
        self.set_cat(person[1])
        self.set_comb_cat(person[2])
        self.set_cont(person[3])
        self.set_comb_cont(person[4])
        self.check_cont()

    def set_imsize(self, image_size):
        image_size = np.array(image_size).flatten().tolist()[0]
        row = np.array(image_size[0]).flatten().tolist()[0]
        col = np.array(image_size[1]).flatten().tolist()[0]
        self.im_size.append(row)
        self.im_size.append(col)

    def validate_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        x1 = min(self.im_size[0], max(0, x1))
        x2 = min(self.im_size[0], max(0, x2))
        y1 = min(self.im_size[1], max(0, y1))
        y2 = min(self.im_size[1], max(0, y2))
        return [int(x1), int(y1), int(x2), int(y2)]

    def set_bbox(self, person_bbox):
        self.bbox = self.validate_bbox(np.array(person_bbox).flatten().tolist())

    def set_cat(self, person_cat):
        self.cat_annotators = len(person_cat[0])
        for ann in range(self.cat_annotators):
            ann_cat = person_cat[0][ann]
            ann_cat = np.array(ann_cat).flatten().tolist()
            ann_cat = np.array(ann_cat[0]).flatten().tolist()
            ann_cat = [np.array(c).flatten().tolist()[0] for c in ann_cat]
            self.cat.append(ann_cat)

    def set_comb_cat(self, person_comb_cat):
        if self.cat_annotators != 0:
            self.comb_cat = [np.array(c).flatten().tolist()[0] for c in person_comb_cat[0]]
        else:
            self.comb_cat = []

    def set_comb_cont(self, person_comb_cont):
        if self.cont_annotators != 0:
            comb_cont = [np.array(c).flatten().tolist()[0] for c in person_comb_cont[0]]
            self.comb_cont = [np.array(c).flatten().tolist()[0] for c in comb_cont[0]]
        else:
            self.comb_cont = []

    def set_cont(self, person_cont):
        self.cont_annotators = len(person_cont[0])
        for ann in range(self.cont_annotators):
            ann_cont = person_cont[0][ann]
            ann_cont = np.array(ann_cont).flatten().tolist()
            ann_cont = np.array(ann_cont[0]).flatten().tolist()
            ann_cont = [np.array(c).flatten().tolist()[0] for c in ann_cont]
            self.cont.append(ann_cont)

    def check_cont(self):
        for c in self.comb_cont:
            if np.isnan(c):
                self.cont_annotators = 0
                break


def cat_to_one_hot(y_cat):
    '''
    One hot encode a categorical label. 
    :param y_cat: Categorical label.
    :return: One hot encoded categorical label. 
    '''
    one_hot_cat = np.zeros(26)
    for em in y_cat:
        one_hot_cat[cat2ind[em]] = 1
    return one_hot_cat

def prepare_data(data_mat, data_path_src, save_dir, dataset_type='train', generate_npy=False, debug_mode=False):
  '''
  Prepare csv files and save preprocessed data in npy files. 
  :param data_mat: Mat data object for a label. 
  :param data_path_src: Path of the parent directory containing the emotic images folders (mscoco, framesdb, emodb_small, ade20k)
  :param save_dir: Path of the directory to save the csv files and the npy files (if generate_npy files is True)
  :param dataset_type: Type of the dataset (train, val or test). Variable used in the name of csv files and npy files. 
  :param generate_npy: If True the data is preprocessed and saved in npy files. Npy files are later used for training. 
  '''
  data_set = list()

  if generate_npy:
    context_arr = list()
    body_arr = list()
    cat_arr = list()
    cont_arr = list()
  
  to_break = 0
  path_not_exist = 0
  cat_cont_zero = 0
  idx = 0
  for ex_idx, ex in enumerate(data_mat[0]):
    nop = len(ex[4][0])
    for person in range(nop):
      if dataset_type == 'train':
        et = emotic_train(ex[0][0],ex[1][0],ex[2],ex[4][0][person])
      else:
        et = emotic_test(ex[0][0],ex[1][0],ex[2],ex[4][0][person])
      try:
        image_path = os.path.join(data_path_src,et.folder,et.filename)
        if not os.path.exists(image_path):
          path_not_exist += 1
          print ('path not existing', ex_idx, image_path)
          continue
        else:
          context = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
          body = context[et.bbox[1]:et.bbox[3],et.bbox[0]:et.bbox[2]].copy()
          context_cv = cv2.resize(context, (224,224))
          body_cv = cv2.resize(body, (128,128))
      except Exception as e:
        to_break += 1
        if debug_mode == True:
            print ('breaking at idx=%d, %d due to exception=%r' %(ex_idx, idx, e))
        continue
      if (et.cat_annotators == 0 or et.cont_annotators == 0):
        cat_cont_zero += 1
        continue
      data_set.append(et)  
      if generate_npy == True:
        context_arr.append(context_cv)
        body_arr.append(body_cv)
        if dataset_type == 'train':
          cat_arr.append(cat_to_one_hot(et.cat))
          cont_arr.append(np.array(et.cont))
        else: 
          cat_arr.append(cat_to_one_hot(et.comb_cat))
          cont_arr.append(np.array(et.comb_cont))
      if idx % 1000 == 0 and debug_mode==False:
        print (" Preprocessing data. Index = ", idx)
      elif idx % 20 == 0 and debug_mode==True:
        print (" Preprocessing data. Index = ", idx)
      idx = idx + 1
    # for debugging purposes
    if debug_mode == True and idx >= 104:
      print (' ######## Breaking data prep step', idx, ex_idx, ' ######')
      print (to_break, path_not_exist, cat_cont_zero)
      cv2.imwrite(os.path.join(save_dir, 'context1.png'), context_arr[-1])
      cv2.imwrite(os.path.join(save_dir, 'body1.png'), body_arr[-1])
      break
  print (to_break, path_not_exist, cat_cont_zero)
  
  csv_path = os.path.join(save_dir, "%s.csv" %(dataset_type))
  with open(csv_path, 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', dialect='excel')
    row = ['Index', 'Folder', 'Filename', 'Image Size', 'BBox', 'Categorical_Labels', 'Continuous_Labels', 'Gender', 'Age']
    filewriter.writerow(row)
    for idx, ex in enumerate(data_set):
        if dataset_type == 'train':
            row = [idx, ex.folder, ex.filename, ex.im_size, ex.bbox, ex.cat, ex.cont, ex.gender, ex.age]
        else:
            row = [idx, ex.folder, ex.filename, ex.im_size, ex.bbox, ex.comb_cat, ex.comb_cont, ex.gender, ex.age]
        filewriter.writerow(row)
  print ('wrote file ', csv_path)

  if generate_npy == True: 
    context_arr = np.array(context_arr)
    body_arr = np.array(body_arr)
    cat_arr = np.array(cat_arr)
    cont_arr = np.array(cont_arr)
    print (len(data_set), context_arr.shape, body_arr.shape)
    np.save(os.path.join(save_dir,'%s_context_arr.npy' %(dataset_type)), context_arr)
    np.save(os.path.join(save_dir,'%s_body_arr.npy' %(dataset_type)), body_arr)
    np.save(os.path.join(save_dir,'%s_cat_arr.npy' %(dataset_type)), cat_arr)
    np.save(os.path.join(save_dir,'%s_cont_arr.npy' %(dataset_type)), cont_arr)
    print (context_arr.shape, body_arr.shape, cat_arr.shape, cont_arr.shape)
  print ('completed generating %s data files' %(dataset_type))
 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Emotic data and annotations')
    parser.add_argument('--save_dir_name', type=str, default='emotic_pre', help='Directory name in which preprocessed data will be stored')
    parser.add_argument('--label', type=str,  default='all', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--generate_npy', action='store_true', help='Generate npy files')
    parser.add_argument('--debug_mode', action='store_true', help='Debug mode. Will only save a small subset of the data')
    # Generate args
    args = parser.parse_args()
    return args
  
if __name__ == '__main__':
    args = parse_args()
    ann_path_src = os.path.join(args.data_dir, 'Annotations','Annotations.mat')
    data_path_src = os.path.join(args.data_dir, 'emotic')
    save_path = os.path.join(args.data_dir, args.save_dir_name)
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    
    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
       'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion
    
    print ('loading Annotations')
    mat = loadmat(ann_path_src)
    if args.label.lower() == 'all':
      labels = ['train', 'val', 'test']
    else:
      labels = [args.label.lower()]
    for label in labels:
      data_mat = mat[label]
      print ('starting label ', label)
      prepare_data(data_mat, data_path_src, save_path, dataset_type=label, generate_npy=args.generate_npy, debug_mode=args.debug_mode)
