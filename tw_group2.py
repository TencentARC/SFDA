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
    
    # finetune acc
    finetune_acc = {
        'aircraft': {'byol': 82.1, 'deepcluster-v2': 82.43, 'infomin': 83.78, 'insdis': 79.7, 'moco-v1': 81.85, 
                    'moco-v2': 83.7, 'pcl-v1': 82.16, 'pcl-v2': 83.0, 'sela-v2': 85.42, 'simclr-v1': 80.54, 
                    'simclr-v2': 81.5, 'swav': 83.04}, 
        'caltech101': {'byol': 91.9, 'deepcluster-v2': 91.16, 'infomin': 80.86, 'insdis': 77.21, 'moco-v1': 79.68, 
                    'moco-v2': 82.76, 'pcl-v1': 88.6, 'pcl-v2': 87.52, 'sela-v2': 90.53, 'simclr-v1': 90.94, 
                    'simclr-v2': 88.58, 'swav': 89.49}, 
        'cars': {'byol': 89.83, 'deepcluster-v2': 90.16, 'infomin': 86.9, 'insdis': 80.21, 'moco-v1': 82.19, 
                    'moco-v2': 85.55, 'pcl-v1': 87.15, 'pcl-v2': 85.56, 'sela-v2': 89.85, 'simclr-v1': 89.98, 
                    'simclr-v2': 88.82, 'swav': 89.81}, 
        'cifar10': {'byol': 96.98, 'deepcluster-v2': 97.17, 'infomin': 96.72, 'insdis': 93.08, 'moco-v1': 94.15, 
                    'moco-v2': 96.48, 'pcl-v1': 96.42, 'pcl-v2': 96.55, 'sela-v2': 96.85, 'simclr-v1': 97.09, 
                    'simclr-v2': 96.22, 'swav': 96.81}, 
        'cifar100': {'byol': 83.86, 'deepcluster-v2': 84.84, 'infomin': 70.89, 'insdis': 69.08, 'moco-v1': 71.23, 
                    'moco-v2': 71.27, 'pcl-v1': 79.44, 'pcl-v2': 79.84, 'sela-v2': 84.36, 'simclr-v1': 84.49, 
                    'simclr-v2': 78.91, 'swav': 83.78}, 
        'dtd': {'byol': 76.37, 'deepcluster-v2': 77.31, 'infomin': 73.47, 'insdis': 66.4, 'moco-v1': 67.36, 
                    'moco-v2': 72.56, 'pcl-v1': 73.28, 'pcl-v2': 69.3, 'sela-v2': 76.03, 'simclr-v1': 73.97, 
                    'simclr-v2': 74.71, 'swav': 76.68}, 
        'flowers': {'byol': 96.8, 'deepcluster-v2': 97.05, 'infomin': 95.81, 'insdis': 93.63, 'moco-v1': 94.32, 
                    'moco-v2': 95.12, 'pcl-v1': 95.62, 'pcl-v2': 95.87, 'sela-v2': 96.22, 'simclr-v1': 95.33, 
                    'simclr-v2': 95.39, 'swav': 97.11}, 
        'food': {'byol': 85.44, 'deepcluster-v2': 87.24, 'infomin': 78.82, 'insdis': 76.47, 'moco-v1': 77.21, 
                    'moco-v2': 77.15, 'pcl-v1': 77.7, 'pcl-v2': 80.29, 'sela-v2': 86.37, 'simclr-v1': 82.2, 
                    'simclr-v2': 82.23, 'swav': 87.22}, 
        'pets': {'byol': 91.48, 'deepcluster-v2': 90.89, 'infomin': 90.92, 'insdis': 84.58, 'moco-v1': 85.26, 
                    'moco-v2': 89.06, 'pcl-v1': 88.93, 'pcl-v2': 88.72, 'sela-v2': 89.61, 'simclr-v1': 88.53, 
                    'simclr-v2': 89.18, 'swav': 90.59}, 
        'sun397': {'byol': 63.69, 'deepcluster-v2': 66.54, 'infomin': 57.67, 'insdis': 51.62, 'moco-v1': 53.83, 
                    'moco-v2': 56.28, 'pcl-v1': 58.36, 'pcl-v2': 58.82, 'sela-v2': 65.74, 'simclr-v1': 63.46, 
                    'simclr-v2': 60.93, 'swav': 66.1}, 
        'voc2007': {'byol': 85.13, 'deepcluster-v2': 85.38, 'infomin': 81.41, 'insdis': 76.33, 'moco-v1': 77.94, 
                    'moco-v2': 78.32, 'pcl-v1': 81.91, 'pcl-v2': 81.85, 'sela-v2': 85.52, 'simclr-v1': 83.29, 
                    'simclr-v2': 83.08, 'swav': 85.06}
        }
    dset = args.dataset
    metric = args.method
    score_path = './results_metrics_cr/group2/{}/{}_metrics.json'.format(metric, dset)
    score, _ = load_score(score_path)
    tw_sfda = w_kendall_metric(score, finetune_acc, dset)
    print("Kendall  dataset:{:12s} SFDA:{:2.3f}".format(dset, tw_sfda))
