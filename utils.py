
import os
import sys
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn import metrics
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score

import numpy as np
from scipy import sparse
import pickle
import pandas as pd
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader, ClusterData, ClusterLoader

from model import get_graph, stDGCC_train, PCA_process, Kmeans_cluster, stDGCC_Model

rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath + '/stDGCC')


def get_data(args):
    data_file = args.data_path + args.data_name + '/'
    with open(data_file + 'euclidean_'+str(args.threshold)+'_Adjacent', 'rb') as fp:
        adj_0 = pickle.load(fp)
    X_data = np.load(data_file + str(args.HVG)+'_HVG_features.npy')

    num_points = X_data.shape[0]

    row = np.arange(0,num_points,1)
    col = np.arange(0,num_points,1)
    data = np.ones(num_points)
    
    adj_I = sparse.csr_matrix((data, (row, col)), shape=(num_points, num_points))

    adj = (1 - args.lambda_I) * adj_0 + args.lambda_I * adj_I

    return adj_0, adj, X_data


def stDGCC(args):
    lambda_I = args.lambda_I
    # Parameters
    batch_size = 1  # Batch size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:',device)
    adj_0, adj, X_data = get_data(args)

    num_cell = X_data.shape[0]
    num_feature = X_data.shape[1]
    print('Adj:', adj.shape, 'Edges:', len(adj.data))
    print('X:', X_data.shape)
    
    n_clusters = args.n_clusters
    print('n clusters:', n_clusters)
    
    if args.DGI and (lambda_I >= 0):
        print("-----------Deep Graph Infomax-------------")
        data_list = get_graph(adj, X_data)
        if args.platform in ['10x','Stereo-seq']:
            data_loader = DataLoader(data_list, batch_size=batch_size)
        elif args.platform in ['Slide-seqV2']:
            data_loader = DataLoader(data_list, batch_size=batch_size)
            for d in data_loader:
                print(d)

        model = stDGCC_train(args, data_loader=data_loader, in_channels=num_feature)
        model.eval()

        for data in data_loader:
            data.to(device)
            X_embedding, _, _, _, _, _ = model(data)
            
            X_embedding = X_embedding.cpu().detach().numpy()

            
            X_embedding_filename = args.embedding_data_path + 'lambdaI' + str(lambda_I) + '_epoch' + str(
                args.num_epoch) + '_Embed_X.npy'
            np.save(X_embedding_filename, X_embedding)

    # 聚类
    if args.cluster:
        cluster_type = 'kmeans'  # 'louvain' leiden kmeans
        print("-----------Clustering-------------")
        es = [args.num_epoch]
        for e in es:
            print(e)
            X_embedding_filename = args.embedding_data_path + 'lambdaI' + str(lambda_I) + '_epoch' + str(e) + '_Embed_X.npy'
            X_embedding = np.load(X_embedding_filename)
            X_embedding = PCA_process(X_embedding, nps=30)
            print(X_embedding.shape)
            print('Shape of data to cluster:', X_embedding.shape)
            if cluster_type == 'kmeans':
                cluster_labels, score = Kmeans_cluster(X_embedding, n_clusters)
                # print(cluster_labels)
                all_data = []
                for index in range(num_cell):
                    all_data.append([index, cluster_labels[index]])  # txt: cell_id, cluster type

                # print(np.array(all_data))
                np.savetxt(args.result_path + '/'+str(e)+'_'+'_types.txt', np.array(all_data), fmt='%3d', delimiter='\t')
                