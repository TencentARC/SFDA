#!/usr/bin/env python
# coding: utf-8

from rank_correlation import (load_score, recall_k, rel_k, pearson_coef, 
                            wpearson_coef, w_kendall_metric, kendall_metric)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate transferability metrics.')
    parser.add_argument('-d', '--dataset', type=str, default='deepcluster-v2', 
                        help='name of the pretrained model to load and evaluate')
    parser.add_argument('-me', '--method', type=str, default='logme', 
                        help='name of used transferability metric')
    args = parser.parse_args()

    finetune_acc = {
        'aircraft': {'resnet34': 84.06, 'resnet50': 84.64, 'resnet101': 85.53, 'resnet152': 86.29, 'densenet121': 84.66, 
                    'densenet169': 84.19, 'densenet201': 85.38, 'mnasnet1_0': 66.48, 'mobilenet_v2': 79.68, 
                    'googlenet': 80.32, 'inception_v3': 80.15}, 
        'caltech101': {'resnet34': 91.15, 'resnet50': 91.98, 'resnet101': 92.38, 'resnet152': 93.1, 'densenet121': 91.5, 
                    'densenet169': 92.51, 'densenet201': 93.14, 'mnasnet1_0': 89.34, 'mobilenet_v2': 88.64, 
                    'googlenet': 90.85, 'inception_v3': 92.75}, 
        'cars': {'resnet34': 88.63, 'resnet50': 89.09, 'resnet101': 89.47, 'resnet152': 89.88, 'densenet121': 89.34, 
                    'densenet169': 89.02, 'densenet201': 89.44, 'mnasnet1_0': 72.58, 'mobilenet_v2': 86.44, 
                    'googlenet': 87.76, 'inception_v3': 87.74}, 
        'cifar10': {'resnet34': 96.12, 'resnet50': 96.28, 'resnet101': 97.39, 'resnet152': 97.53, 'densenet121': 96.45, 
                    'densenet169': 96.77, 'densenet201': 97.02, 'mnasnet1_0': 92.59, 'mobilenet_v2': 94.74, 
                    'googlenet': 95.54, 
                    'inception_v3': 96.18}, 
        'cifar100': {'resnet34': 81.94, 'resnet50': 82.8, 'resnet101': 84.88, 'resnet152': 85.66, 'densenet121': 82.75, 
                    'densenet169': 84.26, 'densenet201': 84.88, 'mnasnet1_0': 72.04, 'mobilenet_v2': 78.11, 
                    'googlenet': 79.84, 
                    'inception_v3': 81.49}, 
        'dtd': {'resnet34': 72.96, 'resnet50': 74.72, 'resnet101': 74.8, 'resnet152': 76.44, 'densenet121': 74.18, 
                    'densenet169': 74.72, 'densenet201': 76.04, 'mnasnet1_0': 70.12, 'mobilenet_v2': 71.72, 
                    'googlenet': 72.53, 
                    'inception_v3': 72.85}, 
        'flowers': {'resnet34': 95.2, 'resnet50': 96.26, 'resnet101': 96.53, 'resnet152': 96.86, 'densenet121': 97.02, 
                    'densenet169': 97.32, 'densenet201': 97.1, 'mnasnet1_0': 95.39, 'mobilenet_v2': 96.2, 
                    'googlenet': 95.76, 
                    'inception_v3': 95.73},
        'food': {'resnet34': 81.99, 'resnet50': 84.45, 'resnet101': 85.58, 'resnet152': 86.28, 'densenet121': 84.99, 
                    'densenet169': 85.84, 'densenet201': 86.71, 'mnasnet1_0': 71.35, 'mobilenet_v2': 81.12, 
                    'googlenet': 79.3, 
                    'inception_v3': 81.76}, 
        'pets': {'resnet34': 93.5, 'resnet50': 93.88, 'resnet101': 93.92, 'resnet152': 94.42, 'densenet121': 93.07, 
                    'densenet169': 93.62, 'densenet201': 94.03, 'mnasnet1_0': 91.08, 'mobilenet_v2': 91.28, 
                    'googlenet': 91.38, 
                    'inception_v3': 92.14},
        'sun397': {'resnet34': 61.02, 'resnet50': 63.54, 'resnet101': 63.76, 'resnet152': 64.82, 'densenet121': 63.26, 
                    'densenet169': 64.1, 'densenet201': 64.57, 'mnasnet1_0': 56.56, 'mobilenet_v2': 60.29, 
                    'googlenet': 59.89, 
                    'inception_v3': 59.98}, 
        'voc2007': {'resnet34': 84.6, 'resnet50': 85.8, 'resnet101': 85.68, 'resnet152': 86.32, 'densenet121': 85.28, 
                    'densenet169': 85.77, 'densenet201': 85.67, 'mnasnet1_0': 81.06, 'mobilenet_v2': 82.8, 
                    'googlenet': 82.58, 
                    'inception_v3': 83.84}
        }
    
    dset = args.dataset
    metric = args.method
    score_path = './results_metrics_cr/group1/{}/{}_metrics.json'.format(metric, dset)
    score, _ = load_score(score_path)
    tw_sfda = w_kendall_metric(score, finetune_acc, dset)
    print("Kendall  dataset:{:12s} SFDA:{:2.3f}".format(dset, tw_sfda))
