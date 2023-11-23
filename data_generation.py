import pandas as pd
import scanpy as sc
import numpy as np
import stlearn as st
import anndata as ad
import random
import matplotlib
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from h5py import Dataset, Group


rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath + '/stDGCC')
# Setting the random number seed
def seed_everything(seed=9527):
    print("seed:"+str(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def adata_preprocess_1(adata, min_cells=50, pca_n_comps=2000, HVG=3000):
    print('===== 1 - Preprocessing Data ')
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=HVG)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    # adata_X = sc.pp.pca(adata.X, n_comps=pca_n_comps)
    # adata_X = sc.pp.pca(adata[:, adata.var['highly_variable']].X, n_comps=pca_n_comps)
    return adata[:, adata.var['highly_variable']].X

def adata_preprocess_2(adata, min_cells=5, pca_n_comps=1500, HVG=3000):
    print('===== 2 - Preprocessing Data ')
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=HVG)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    adata_X = sc.pp.pca(adata[:, adata.var['highly_variable']].X, n_comps=pca_n_comps)

    return adata_X
def adata_preprocess_3(adata, min_cells=50, pca_n_comps=2000, HVG=3000):
    print('===== 3 - Preprocessing Data ')
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(adata)
    datasets=[adata[adata.obs.batch=="0"].X, adata[adata.obs.batch=="1"].X,adata[adata.obs.batch=="2"].X]
    genes_list = list(adata.var_names)
    features_corrected = batch_correction(datasets, genes_list)
    adata.X = features_corrected
    print(adata.X)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=HVG)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # print(adata.X.shape)
    # adata.write('/home/zhangyingxi/stDGCC_AAAI/stDGCC/generated_data/merfish/merfish.h5ad')
    # adata_X = sc.pp.pca(adata.X, n_comps=pca_n_comps)
    return adata[:, adata.var['highly_variable']].X
def batch_correction(datasets, genes_list):
    # List of datasets (matrices of cells-by-genes):
    #datasets = [ list of scipy.sparse.csr_matrix or numpy.ndarray ]
    # List of gene lists:
    genes_lists = [ genes_list, genes_list, genes_list ]

    import scanorama

    # Batch correction.
    corrected, _ = scanorama.correct(datasets, genes_lists)
    features_corrected = []
    for i, corrected_each_batch in enumerate(corrected):
        #features_corrected.append(np.array(corrected_each_batch.A))
        if i == 0:
            features_corrected = corrected_each_batch.A
        else:
            features_corrected = np.vstack((features_corrected, corrected_each_batch.A))
    features_corrected = np.array(features_corrected)
#     np.save( generated_data_path + 'features.npy', features_corrected)
    print('corrected size: ', features_corrected.shape)
    return features_corrected
# Construct an adjacency matrix and store to generated_data_fold
def adj_radius(dis, t, generated_data_fold):
    from scipy import sparse
    import pickle

    coordinates = np.load(generated_data_fold + 'coordinates.npy')
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(radius = t, metric=dis).fit(coordinates)

    distances, indices = nbrs.radius_neighbors(coordinates, return_distance=True)

    adj = np.zeros([distances.shape[0],distances.shape[0]])
    for node_idx in range(indices.shape[0]):
        adj[node_idx][indices[node_idx]] = 1
    adj = adj - np.eye(distances.shape[0])

    with open(generated_data_fold +dis+'_'+str(t)+'_Adjacent', 'wb') as fp:
        pickle.dump(sparse.csr_matrix(adj), fp)

def adj_radius_batch(dis, t, batch, coordinates, generated_data_fold):
    from scipy import sparse
    import pickle

    # coordinates = np.load(generated_data_fold + 'coordinates.npy')
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(radius = t, metric=dis).fit(coordinates)

    distances, indices = nbrs.radius_neighbors(coordinates, return_distance=True)

    adj = np.zeros([distances.shape[0],distances.shape[0]])
    for node_idx in range(indices.shape[0]):
        adj[node_idx][indices[node_idx]] = 1
    adj = adj - np.eye(distances.shape[0])

    with open(generated_data_fold +dis+'_'+str(t)+'_Adjacent_'+str(batch), 'wb') as fp:
        pickle.dump(sparse.csr_matrix(adj), fp)
    return adj

def get_Slide_seqV2_MH(count_file,pos_file):
    adata_h5 = sc.read_text(count_file, delimiter='\t')
    adata_h5 = ad.AnnData(adata_h5.X.T, var =adata_h5.obs ,obs=adata_h5.var)
    coordinates = pd.read_csv(pos_file, index_col=0)
    spatial = coordinates.loc[adata_h5.obs_names, ['xcoord', 'ycoord']].to_numpy()
    adata_h5.var_names_make_unique()
    adata_h5.obsm['spatial'] = spatial
    return adata_h5

def get_Slide_seqV1_MH(count_file,pos_file):
    adata_h5 = sc.read_csv(count_file)
    adata_h5 = ad.AnnData(adata_h5.X.T, var =adata_h5.obs ,obs=adata_h5.var)
    coordinates = pd.read_csv(pos_file, index_col=0)
    spatial = coordinates.loc[adata_h5.obs_names, ['xcoord', 'ycoord']].to_numpy()
    adata_h5.var_names_make_unique()
    adata_h5.obsm['spatial'] = spatial
    print(adata_h5)
    return adata_h5


def get_MERFISH(count_file, pos_file):
    df = pd.read_csv(count_file, index_col=0).T
# coordinates字典，
    coordinates = pd.read_excel(pos_file, index_col=0, sheet_name=[0,1,2])
    merge = pd.concat(coordinates.values())
    adata_h5 = ad.AnnData(df)
    df = pd.DataFrame(index= merge.index, columns = ["batch"])
    df.iloc[:coordinates[0].shape[0]] = "0"
    df.iloc[coordinates[0].shape[0]:coordinates[0].shape[0]+coordinates[1].shape[0]] = "1"
    df.iloc[-coordinates[2].shape[0]:] = "2"
    adata_h5.obsm['spatial'] = merge
    adata_h5.obs['batch'] = df
    return adata_h5
def main(args):
    print(args)
    data_fold = args.data_path + args.data_name + '/'
    print(data_fold)
    generated_data_fold = args.generated_data_path + args.data_name + '/'
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold)
    if args.platform=='10x':
        adata_h5 = st.Read10X(path=data_fold, count_file='filtered_feature_bc_matrix.h5')
        adata_h5.var_names_make_unique()
        if args.data_name == 'MB':
            # adata_h5 = sc.read_visium(path=data_fold, count_file='filtered_feature_bc_matrix.h5')
            # adata_h5.var_names_make_unique()
            adata_h5 = adata_h5[(adata_h5.obsm['spatial'][:,1]>2750) & (adata_h5.obsm['spatial'][:,1]<5500) & (adata_h5.obsm['spatial'][:,0]<6500)]
            features = adata_preprocess_1(adata_h5, pca_n_comps=2000, min_cells=args.min_cells, HVG = args.HVG).toarray()
        if args.data_name != 'MB':
            features = adata_preprocess_2(adata_h5, min_cells=args.min_cells, pca_n_comps=1500, HVG = args.HVG)
    elif args.platform=='Stereo-seq':
        adata_h5 = sc.read_h5ad(data_fold+'E13.5_E1S3.MOSTA.h5ad')
        adata_h5.X = adata_h5.layers['count']
        adata_h5.var_names_make_unique()
        features = adata_preprocess_2(adata_h5, min_cells=args.min_cells, pca_n_comps=1500, HVG = args.HVG)
    elif args.platform=='Slide-seqV2':
        adata_h5 = get_Slide_seqV2_MH(count_file=data_fold+'Puck_190921_21.digital_expression.txt',
                                  pos_file=data_fold+'Puck_190921_21_bead_locations.csv')
        features = adata_preprocess_1(adata_h5, pca_n_comps=2000, min_cells=args.min_cells, HVG = args.HVG)  
    elif args.platform=='Slide-seqV1': 
        adata_h5 = get_Slide_seqV1_MH(count_file=data_fold+'MappedDGEForR.csv',
                                  pos_file=data_fold+'BeadLocationsForR.csv')
        features = adata_preprocess_1(adata_h5, pca_n_comps=2000, min_cells=args.min_cells, HVG = args.HVG)         
    elif args.platform=='MERFISH':
        # python data_generation.py --data_path dataset/ --data_name merfish --generated_data_path generated_data/ --platform MERFISH --threshold 200
        adata_h5 = get_MERFISH(count_file=data_fold+'pnas.1912459116.sd12.csv', pos_file=data_fold+'pnas.1912459116.sd15.xlsx')
        features = adata_preprocess_3(adata_h5, pca_n_comps=1500, min_cells=args.min_cells, HVG = args.HVG)
        np.save(generated_data_fold +str(args.HVG)+'_HVG_features.npy', features)
        adjs = []
        for i in range(0,3):
            coordinates = adata_h5[adata_h5.obs.batch==str(i)].obsm['spatial']
            adj = adj_radius_batch(dis='euclidean', t=args.threshold, batch = i, coordinates = coordinates, generated_data_fold=generated_data_fold)
            adjs.append(adj)
        import scipy
        all_adj = scipy.linalg.block_diag(adjs[0], adjs[1], adjs[2])
        import pickle
        from scipy import sparse
        with open(generated_data_fold +'euclidean'+'_'+str(args.threshold)+'_Adjacent', 'wb') as fp:
            pickle.dump(sparse.csr_matrix(all_adj), fp)
        return 0
    

    coordinates = adata_h5.obsm['spatial']
    np.save(generated_data_fold + 'coordinates.npy', np.array(coordinates))
    adj_radius(dis='euclidean',t=args.threshold,generated_data_fold=generated_data_fold)
    print(features)
    np.save(generated_data_fold +str(args.HVG)+'_HVG_features.npy', features)


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--min_cells', type=float, default=5,
                        help='Lowly expressed genes which appear in fewer than this number of cells will be filtered out')
    parser.add_argument('--data_path', type=str, default='dataset/DLPFC/', help='The path to dataset')
    parser.add_argument('--data_name', type=str, default=' ',
                        help='The name of dataset')
    parser.add_argument('--generated_data_path', type=str, default='generated_data/DLPFC/',
                        help='The folder to store the generated data')
    parser.add_argument('--seed', type=int, default=317, help='Seed')
    parser.add_argument('--HVG', type=int, default=3000, help='Number of highly variable genes')
    parser.add_argument('--threshold', type=int, default=250, help='Threshold for constructing the adjacency matrix')
    parser.add_argument('--platform', type=str, default='10x', help='Platform for dataset')

    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)

