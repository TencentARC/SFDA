#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import torch.nn as nn
import models.group1 as models


import numpy as np

from utils import forward_pass_feature
from get_dataloader import prepare_data, get_data


class Pretrained_ResNet50(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.resnet50(pretrained=False)
        del self.model.fc

        # download the ckpt to the following folder
        state_dict = torch.load(os.path.join('models/group2', self.model_name + '.pth'))
        self.model.load_state_dict(state_dict)
        print("Model {} checkpoint loaded".format(model_name))

        self.model.train()
        print("num parameters:", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract feature for self-supervised models.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', 
                        help='name of the dataset to evaluate on')
    parser.add_argument('-b', '--batch-size', type=int, default=64, 
                        help='the size of the mini-batches when inferring features')
    parser.add_argument('-i', '--image-size', type=int, default=224, 
                        help='the size of the input images')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')
    args = parser.parse_args()
    args.norm = not args.no_norm
    pprint(args)

    # load dataset
    dset, data_dir, num_classes, metric = get_data(args.dataset)
    args.num_classes = num_classes
    
    train_loader, val_loader, trainval_loader, test_loader, all_loader = prepare_data(
        dset, data_dir, args.batch_size, args.image_size, normalisation=args.norm)
    
    print(f'Train:{len(train_loader.dataset)}, Val:{len(val_loader.dataset)},' 
            f'TrainVal:{len(trainval_loader.dataset)}, Test:{len(test_loader.dataset)} '
            f'AllData:{len(all_loader.dataset)}')
    
    trainval_loader = all_loader

    # load pretrained model and forward features
    score_dict = {}   
    duration = 0

    fpath = os.path.join('./results_f', 'group2')
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    models_hub = ['byol', 'deepcluster-v2', 'infomin', 'insdis', 'moco-v1',
              'moco-v2', 'pcl-v1', 'pcl-v2', 'sela-v2', 'simclr-v1', 
              'simclr-v2', 'swav']
    for model in models_hub:
        args.model = model
        model_npy_feature = os.path.join(fpath, f'{args.model}_{args.dataset}_feature.npy')
        model_npy_label = os.path.join(fpath, f'{args.model}_{args.dataset}_label.npy')

        if os.path.exists(model_npy_feature) and os.path.exists(model_npy_label):
            print(f"Features and Labels of {args.model} on {args.dataset} has been saved.")
            continue

        
        model = Pretrained_ResNet50(args.model)
        model = model.to(args.device)
        X_trainval_feature, y_trainval = forward_pass_feature(trainval_loader, model)   
        if args.dataset == 'voc2007':
            y_trainval = torch.argmax(y_trainval, dim=1)
        print(f'x_trainval shape:{X_trainval_feature.shape} and y_trainval shape:{y_trainval.shape}')
        
        np.save(model_npy_feature, X_trainval_feature.numpy())
        np.save(model_npy_label, y_trainval.numpy())
        print(f"Features and Labels of {args.model} on {args.dataset} has been saved.")
    