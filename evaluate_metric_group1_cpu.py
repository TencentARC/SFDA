#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import models.group1 as models
import numpy as np

import json
import time

from metrics import LEEP, NLEEP, LogME_Score, SFDA_Score, PARC_Score


def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict 
        json.dump(score_dict, f)


def exist_score(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transferability score.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', 
                        help='name of the dataset to evaluate on')
    parser.add_argument('-me', '--metric', type=str, default='logme', 
                        help='name of the method for measuring transferability')   
    parser.add_argument('--nleep-ratio', type=float, default=5, 
                        help='the ratio of the Gaussian components and target data classess')
    parser.add_argument('--parc-ratio', type=float, default=2,
                        help='PCA reduction dimension')
    parser.add_argument('--output-dir', type=str, default='./results_metrics/group1', 
                        help='dir of output score')
    args = parser.parse_args()   
    pprint(args)

    score_dict = {}   
    fpath = os.path.join(args.output_dir, args.metric)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fpath = os.path.join(fpath, f'{args.dataset}_metrics.json')

    if not os.path.exists(fpath):
        save_score(score_dict, fpath)
    else:
        with open(fpath, "r") as f:
            score_dict = json.load(f)
    
    models_hub = ['inception_v3', 'mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201', 
                    'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']
    for model in models_hub:
        if exist_score(model, fpath):
            print(f'{model} has been calculated')
            continue
        args.model = model
        
        model_npy_feature = os.path.join('./results_f/group1', f'{args.model}_{args.dataset}_feature.npy')
        model_npy_label = os.path.join('./results_f/group1', f'{args.model}_{args.dataset}_label.npy')
        X_features, y_labels = np.load(model_npy_feature), np.load(model_npy_label)

        print(f'x_trainval shape:{X_features.shape} and y_trainval shape:{y_labels.shape}')        
        print(f'Calc Transferabilities of {args.model} on {args.dataset}')
     
        if args.metric == 'logme':
            score_dict[args.model] = LogME_Score(X_features, y_labels)
        elif args.metric == 'leep':     
            score_dict[args.model] = LEEP(X_features, y_labels, model_name=args.model)
        elif args.metric == 'parc':           
            score_dict[args.model] = PARC_Score(X_features, y_labels, ratio=args.parc_ratio)
        elif args.metric == 'nleep':           
            ratio = 1 if args.dataset in ('food', 'pets') else args.nleep_ratio
            score_dict[args.model] = NLEEP(X_features, y_labels, component_ratio=ratio)
        elif args.metric == 'sfda':
            score_dict[args.model] = SFDA_Score(X_features, y_labels)   
        else:
            raise NotImplementedError
        
        print(f'{args.metric} of {args.model}: {score_dict[args.model]}\n')
        save_score(score_dict, fpath)
        
    results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
    print(f'Models ranking on {args.dataset} based on {args.metric}: ')
    pprint(results)
    results = {a[0]: a[1] for a in results}
    save_score(results, fpath)
