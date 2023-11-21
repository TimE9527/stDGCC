import itertools
import os
import sys
import matplotlib
from torch_geometric.nn.inits import reset, uniform

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics
from scipy import sparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader

EPS = 1e-15
MAX_LOGSTD = 20


def get_graph(adj, X):

    row_col = []
    edge_weight = []

    rows, cols = adj.nonzero()

    edge_nums = adj.getnnz()
    for i in range(edge_nums):
        row_col.append([rows[i], cols[i]])
        edge_weight.append(adj.data[i])
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    graph_bags = []
    graph = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    graph_bags.append(graph)
    return graph_bags




# encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        # 1024
        h1 = hidden_channels * 8
        # 512
        h2 = hidden_channels * 4
        # 256
        h3 = hidden_channels * 2
        # 128
        h4 = hidden_channels
        self.conv = GCNConv(in_channels, h1)
        self.conv_2 = GCNConv(h1, h2)
        self.conv_3 = GCNConv(h2, h3)
        # self.conv_4 = GCNConv(h3, h4)
  
        self.conv_mu = GCNConv(h3, h4)
        self.conv_logvar = GCNConv(h3, h4)
        self.prelu = nn.PReLU(h4)
        print(h1)
        print(h4)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.conv_2(x, edge_index, edge_weight=edge_weight)
        x = self.conv_3(x, edge_index, edge_weight=edge_weight)
        # vgcn
        mu = self.conv_mu(x, edge_index, edge_weight=edge_weight)
        logvar = self.conv_logvar(x, edge_index, edge_weight=edge_weight)
        z = self.reparameterize(mu,logvar)

        # z = self.conv_4(x, edge_index, edge_weight=edge_weight)
        #
        z = self.prelu(z)
        return z, mu, logvar
    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(max=MAX_LOGSTD)
        if self.training:
            # print("traning")
            return mu + torch.randn_like(logvar) * torch.exp(0.5*logvar)
        else:
            # print("not traning")
            return mu

class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_channels):
        super(Decoder, self).__init__()
        # 1024
        h1 = hidden_channels * 8
        # 512
        h2 = hidden_channels * 4
        # 256
        h3 = hidden_channels * 2
        # 128
        h4 = hidden_channels
        self.dense_1 = nn.Linear(h4, h3)
        self.dense_2 = nn.Linear(h3, h2)
        self.dense_3 = nn.Linear(h2, h1)
        self.dense_4 = nn.Linear(h1, out_channels)
        self.sigmod = nn.Sigmoid()
    def forward(self, data):
        x = self.dense_1(data)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        # x = self.sigmod(x)
        return x


class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
# corruption function
def Corruption(data):
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)
class stDGCC_Model(nn.Module):
    def __init__(self, hidden_channels, encoder, summary, corruption, decoder):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.encoder = encoder

        self.decoder = decoder

        self.corruption = corruption

        self.summary = summary
        self.weight = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))

        self.reset_parameters()
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        reset(self.decoder)
        uniform(self.hidden_channels, self.weight)

    def forward(self, data):
        pos_z, mu, logvar = self.encoder(data)
        cor = self.corruption(data)
        cor = cor if isinstance(cor, tuple) else (cor,)
        neg_z, _, _ = self.encoder(*cor)

        s = self.summary(pos_z)

        x_ = self.decoder(pos_z)

        return pos_z, neg_z, s, mu, logvar, x_

    def discriminate(self, z, summary, sigmoid=True):
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def CL_Loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        Cos_loss = -torch.log(1 - F.cosine_similarity(pos_z,neg_z) + EPS).mean()
        return pos_loss + neg_loss + Cos_loss

    def KL_Loss(self, mu, logvar):
        # print(1)
        logvar = logvar.clamp(max=MAX_LOGSTD)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        return KLD
    def Reconstructio_Loss(self, x_, x):
        MSE = F.mse_loss(x_, x, reduction='sum') / x.shape[0]
        return MSE

def stDGCC_train(args, data_loader, in_channels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = stDGCC_Model(
        hidden_channels=args.hidden,
        encoder=Encoder(in_channels=in_channels, hidden_channels=args.hidden),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=Corruption,
        decoder = Decoder(out_channels=in_channels, hidden_channels=args.hidden)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.load:
        filename = args.model_path + 'DGI_lambdaI_' + str(args.lambda_I) + '_epoch' + str(
            args.num_epoch) + '.pth.tar'
        model.load_state_dict(torch.load(filename))
    else:
        import datetime
        start_time = datetime.datetime.now()

        for epoch in range(args.num_epoch):
            model.train()
            optimizer.zero_grad()
            CL_loss_v = []
            MSE_loss_v = []
            KL_loss_v = []
            Cos_loss_v = []
            All_loss_v = []
            for data in data_loader:
                data = data.to(device)
                pos_z, neg_z, summary, mu, logvar, x_ = model(data=data)

                CL_loss = model.CL_Loss(pos_z, neg_z, summary)
                MSE_loss = model.Reconstructio_Loss(x_, data.x)
                KL_loss = model.KL_Loss(mu, logvar)



                loss = args.DGI_P *CL_loss + args.MSE_P * MSE_loss + args.KL_P * KL_loss

                loss.backward()

                All_loss_v.append(loss.item())
                CL_loss_v.append(CL_loss.item())
                MSE_loss_v.append(MSE_loss.item())
                KL_loss_v.append(KL_loss.item())
  
                optimizer.step()

            if ((epoch + 1) % 200) == 0:
                # print(data.x)
                print('Epoch: {:03d}, Loss: {:.4f}, CL_Loss: {:.4f}, MSE_Loss: {:.4f}, KL_Loss: {:.4f}'
                      .format(epoch + 1, np.mean(All_loss_v), np.mean(CL_loss_v), np.mean(MSE_loss_v), np.mean(KL_loss_v)))
                
            # if (epoch+1) in [50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 14500, 15000, 15500,16000, 17000, 18000, 19000,20000,21000,22000,23000,24000,25000]:
            #     print(epoch)
            #     DGI_filename = args.model_path + 'DGI_lambdaI_' + str(args.lambda_I) + '_epoch' + str(epoch+1) + '.pth.tar'
            #     torch.save(model.state_dict(), DGI_filename)

            
                

        end_time = datetime.datetime.now()
        print('Training time in seconds: ', (end_time - start_time).seconds)

        print(epoch)
        DGI_filename = args.model_path + 'DGI_lambdaI_' + str(args.lambda_I) + '_epoch' + str(
            args.num_epoch) + '.pth.tar'
        torch.save(model.state_dict(), DGI_filename)
    return model




def PCA_process(X, nps):
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)  
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC



from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation


def Kmeans_cluster(X_embedding, n_clusters, merge=False):
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(X_embedding)

    score = metrics.silhouette_score(X_embedding, cluster_labels, metric='euclidean')

    return cluster_labels, score
