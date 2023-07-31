
import os
import random
import sys
import matplotlib
from utils import stDGCC

matplotlib.use('Agg')

import matplotlib.pyplot as plt


from sklearn import metrics
from scipy import sparse


import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader
from datetime import datetime
import stlearn as st
rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath + '/stDGCC')
def seed_everything(seed=317):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # =========================== args ===============================
    parser.add_argument('--data_name', type=str, default='151507',
                        help="the name of dataset")
    parser.add_argument('--lambda_I', type=float, default=0.8)
    parser.add_argument('--data_path', type=str, default='generated_data/DLPFC/', help='data path')
    parser.add_argument('--model_path', type=str, default='model/DLPFC/')
    parser.add_argument('--embedding_data_path', type=str, default='embedding/DLPFC/')
    parser.add_argument('--result_path', type=str, default='result/DLPFC/')
    parser.add_argument('--DGI', type=int, default=1,
                        help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings')
    parser.add_argument('--load', type=int, default=0, help='Load pretrained DGI model')
    parser.add_argument('--num_epoch', type=int, default=2500, help='numebr of epoch in training DGI')
    parser.add_argument('--hidden', type=int, default=128, help='hidden channels in DGI')
    parser.add_argument('--PCA', type=int, default=1, help='run PCA or not')
    parser.add_argument('--cluster', type=int, default=1, help='run cluster or not')
    parser.add_argument('--n_clusters', type=int, default=7,
                        help='number of clusters in Kmeans, when ground truth label is not avalible.')  # 5 on MERFISH, 20 on Breast
    parser.add_argument('--DGI_P', type=float, default=2.0)
    parser.add_argument('--KL_P', type=float, default=0.005)
    parser.add_argument('--MSE_P', type=float, default=0.05)
    parser.add_argument('--COS_P', type=float, default=2.0)
    parser.add_argument('--HVG', type=int, default=3000, help='Number of highly variable genes')
    parser.add_argument('--threshold', type=int, default=250, help='Threshold for constructing the adjacency matrix')
    parser.add_argument('--platform', type=str, default='10x', help='Platform for dataset')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    seed_everything(317)
    args = parser.parse_args()
    args.COS_P = args.DGI_P

    # file_name = 'all'+'/'+str(args.DGI_P)+'_'+str(args.MSE_P)+'_'+str(args.KL_P) + '/'
    args.model_path = args.model_path + args.data_name + '/'
    args.result_path = args.result_path + args.data_name + '/'
    args.embedding_data_path = args.embedding_data_path + args.data_name + '/'
    args.result_path = args.result_path + 'lambdaI' + str(args.lambda_I) + '/'
    if not os.path.exists(args.embedding_data_path):
        os.makedirs(args.embedding_data_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(args.embedding_data_path):
        os.makedirs(args.embedding_data_path)
    print('------------------------Model and Training Details--------------------------')
    print(args)
    stDGCC(args)




