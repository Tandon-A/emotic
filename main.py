import argparse
import os

from emotic import Emotic
from train import train_emotic
from test import test_emotic
from inference import inference_emotic

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--mode', type=str, default='train_test', choices=['train', 'test', 'train_test', 'inference'])
    parser.add_argument('--data_path', type=str, help='Path to preprocessed data npy files/ csv files')
    parser.add_argument('--experiment_path', type=str, required=True, help='Path to save experiment files (results, models, logs)')
    parser.add_argument('--model_dir_name', type=str, default='models', help='Name of the directory to save models')
    parser.add_argument('--result_dir_name', type=str, default='results', help='Name of the directory to save results(predictions, labels mat files)')
    parser.add_argument('--log_dir_name', type=str, default='logs', help='Name of the directory to save logs (train, val)')
    parser.add_argument('--inference_file', type=str, help='Text file containing image context paths and bounding box')
    parser.add_argument('--context_model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='context model type')
    parser.add_argument('--body_model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='body model type')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--cat_loss_weight', type=float, default=0.5, help='weight for discrete loss')
    parser.add_argument('--cont_loss_weight', type=float, default=0.5, help='weight fot continuous loss')
    parser.add_argument('--continuous_loss_type', type=str, default='Smooth L1', choices=['L2', 'Smooth L1'], help='type of continuous loss')
    parser.add_argument('--discrete_loss_weight_type', type=str, default='dynamic', choices=['dynamic', 'mean', 'static'], help='weight policy for discrete loss')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=52) # use batch size = double(categorical emotion classes)
    # Generate args
    args = parser.parse_args()
    return args


def check_paths(args):    
    ''' Check (create if they don't exist) experiment directories.
    :param args: Runtime arguments as passed by the user.
    :return: List containing result_dir_path, model_dir_path, train_log_dir_path, val_log_dir_path.
    '''
    folders= [args.result_dir_name, args.model_dir_name]
    paths = list()
    for folder in folders:
        folder_path = os.path.join(args.experiment_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)
        
    log_folders = ['train', 'val']
    for folder in log_folders:
        folder_path = os.path.join(args.experiment_path, args.log_dir_name, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)
    return paths


if __name__ == '__main__':
    args = parse_args()
    print ('mode ', args.mode)

    result_path, model_path, train_log_path, val_log_path = check_paths(args)

    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
            'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness', \
            'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion
    
    vad = ['Valence', 'Arousal', 'Dominance']
    ind2vad = {}
    for idx, continuous in enumerate(vad):
        ind2vad[idx] = continuous

    context_mean = [0.4690646, 0.4407227, 0.40508908]
    context_std = [0.2514227, 0.24312855, 0.24266963]
    body_mean = [0.43832874, 0.3964344, 0.3706214]
    body_std = [0.24784276, 0.23621225, 0.2323653]
    context_norm = [context_mean, context_std]
    body_norm = [body_mean, body_std]

    if args.mode == 'train':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for training')
        with open(os.path.join(args.experiment_path, 'config.txt'), 'w') as f:
            print(args, file=f)
        train_emotic(result_path, model_path, train_log_path, val_log_path, ind2cat, ind2vad, context_norm, body_norm, args)
    elif args.mode == 'test':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for testing')
        test_emotic(result_path, model_path, ind2cat, ind2vad, context_norm, body_norm, args)
    elif args.mode == 'train_test':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for training and testing')
        with open(os.path.join(args.experiment_path, 'config.txt'), 'w') as f:
            print(args, file=f)
        train_emotic(result_path, model_path, train_log_path, val_log_path, ind2cat, ind2vad, context_norm, body_norm, args)
        test_emotic(result_path, model_path, ind2cat, ind2vad, context_norm, body_norm, args)
    elif args.mode == 'inference':
        if args.inference_file is None:
            raise ValueError('Inference file not provided. Please pass a valid inference file for inference')
        inference_emotic(args.inference_file, model_path, result_path, context_norm, body_norm, ind2cat, ind2vad, args)
    else:
        raise ValueError('Unknown mode')
