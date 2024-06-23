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


def adata_preprocess_4(adata, min_cells=5):
    print('===== 4 - Preprocessing Data ')
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_counts=1)
    print(adata)
    # print(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    # print(adata.X)
    return adata.X


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
    print(adata.X.shape)

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

    # adj = np.zeros([distances.shape[0],distances.shape[0]])
    # for node_idx in range(indices.shape[0]):
    #     adj[node_idx][indices[node_idx]] = 1
    # adj = adj - np.eye(distances.shape[0])
    # adj = sparse.csr_matrix(adj)

    num_nodes = len(indices)
    row_indices = []
    col_indices = []
    for i, neighbor_indices in enumerate(indices):
        row_indices.extend([i] * len(neighbor_indices))
        col_indices.extend(neighbor_indices)
    data = np.ones(len(row_indices))
    adj = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    adj.setdiag(0)

    with open(generated_data_fold +dis+'_'+str(t)+'_Adjacent', 'wb') as fp:
        pickle.dump(adj, fp)

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


def get_MERFISH():
    adata = sc.read_h5ad("dataset/MERFISH/MERFISH_Allen2022Molecular_aging_MsBrainAgingSpatialDonor_4_0_data.h5ad")
    return adata

def get_Xenium():
    adata = sc.read("dataset/Xenium/matrix.mtx.gz")
    adata = adata.T
    location = pd.read_csv("dataset/Xenium/cells.csv.gz")
    location = location[['x_centroid','y_centroid']]
    adata.obsm['spatial'] = location.to_numpy()
    adata = adata[(adata.obsm['spatial'][:,0]>2000) & (adata.obsm['spatial'][:,0]<8000) & (adata.obsm['spatial'][:,1]>3000)]
    adata.var_names_make_unique()
    return adata

def get_CosMX():
    # 38987
    exprMat = pd.read_csv("dataset/CosMX/Quarter Brain/Run5642_S3_Quarter_exprMat_file.csv")
    metadata = pd.read_csv("dataset/CosMX/Quarter Brain/Run5642_S3_Quarter_metadata_file.csv")
    exprMat['spots'] = exprMat['fov'].astype(str) + '_'+exprMat['cell_ID'].astype(str)
    metadata['spots'] = metadata['fov'].astype(str) + '_'+metadata['cell_ID'].astype(str)
    exprMat.set_index('spots', inplace=True)
    metadata.set_index('spots', inplace=True)

    location = metadata[['CenterX_global_px', 'CenterY_global_px']]
    row_names = location.index.tolist()

    count = exprMat
    count =count.iloc[:, 2:]
    count =count.iloc[:, :-10]
    count = count.loc[row_names]


    import anndata as ad
    adata = ad.AnnData(count)
    adata.obsm['spatial'] = location.to_numpy()
    return adata

def get_MERFISH2():
    import pandas as pd
    countMat = pd.read_csv("dataset/MERFISH2/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate1_cell_by_gene_S2R1.csv", index_col=0)
    metadata = pd.read_csv("dataset/MERFISH2/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate1_cell_metadata_S2R1.csv", index_col=0)
    location = metadata.loc[countMat.index.tolist()][['center_x','center_y']]
    import anndata as ad
    adata = ad.AnnData(countMat)
    adata.obsm['spatial'] = location.to_numpy()
    adata = adata[:,0:483]

    return adata


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
        # 读取adata
        adata_h5 = sc.read_h5ad(data_fold+'E13.5_E1S3.MOSTA.h5ad')
        adata_h5.X = adata_h5.layers['count']
        adata_h5.var_names_make_unique()
        # 预处理
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
        adata_h5 = get_MERFISH()
        features = adata_h5.X.toarray()
    elif args.platform=='MERFISH2':
        adata_h5 = get_MERFISH2()
        features = adata_preprocess_1(adata_h5, pca_n_comps=2000, min_cells=args.min_cells, HVG = args.HVG) 


    elif args.platform=='CosMX':
        adata_h5 = get_CosMX()
        features = adata_preprocess_4(adata_h5, min_cells=args.min_cells)
    elif args.platform=='Xenium':
        adata_h5 = get_Xenium()
        # features = adata_preprocess_1(adata_h5, pca_n_comps=2000, min_cells=args.min_cells, HVG = args.HVG).toarray() 
        features = adata_preprocess_1(adata_h5, pca_n_comps=2000, min_cells=args.min_cells, HVG = args.HVG).toarray()  
        print(features.shape)

    
    # 存储坐标，计算graph
    coordinates = adata_h5.obsm['spatial']
    np.save(generated_data_fold + 'coordinates.npy', np.array(coordinates))
    adj_radius(dis='euclidean',t=args.threshold,generated_data_fold=generated_data_fold)
    print(features.shape)
    np.save(generated_data_fold +str(args.HVG)+'_HVG_features.npy', features)


if __name__ == "__main__":
    import argparse
    # CUDA_VISIBLE_DEVICES=2 python run.py --data_path generated_data/ --data_name Xenium --num_epoch 3000 --DGI_P 1.0 --MSE_P 0.05 --KL_P 0.005 --HVG 3000 --threshold 35 --lambda_I 0.2 --n_clusters 13 --learning_rate 1e-5  --platform MERFISH --model_path model/ --embedding_data_path embedding/ --result_path result/
    # python data_generation.py --data_path dataset/ --data_name Xenium --generated_data_path generated_data/ --platform Xenium --threshold 35
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

