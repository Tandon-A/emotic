import numpy as np 
import os 
import cv2

import torch 
from torchvision import transforms

from yolo_utils import prepare_yolo, nms_people, rescale_boxes

def get_bbox(yolo_model, device, image_context_path, bbox = None, conf_thresh=0.8, nms_thresh=0.4):
  image_context =  cv2.cvtColor(cv2.imread(image_context_path), cv2.COLOR_BGR2RGB)
  test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
  image_yolo = test_transform(cv2.resize(image_context, (416, 416))).unsqueeze(0).to(device)

  with torch.no_grad():
    detections = yolo_model(image_yolo)
    print (detections.shape)
    detections = nms_people(detections, conf_thresh, nms_thresh)
    print (len(detections))

  image = cv2.rectangle(image_context, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3) 
  cv2.imwrite('/home/abtandon/proj/data/debug_exp/img.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

  return [0, 0, 0, 0]

def yolo_infer(model_dir, images_list):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  yolo = prepare_yolo(model_dir)
  yolo = yolo.to(device)
  yolo.eval()

  with open(images_list, 'r') as f:
    lines = f.readlines()
  
  for idx, line in enumerate(lines):
    image_context_path, x1, y1, x2, y2 = line.split('\n')[0].split(' ')
    bbox = [int(x1), int(y1), int(x2), int(y2)]
    bbox2 = get_bbox(yolo, device, image_context_path, bbox)
    print ('pred ', bbox2, 'gt', bbox)
    



yolo_infer('/home/abtandon/proj/data/debug_exp/models', '/home/abtandon/proj/data/debug_exp/inference_list.txt')
