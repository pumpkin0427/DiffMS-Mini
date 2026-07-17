import torch
import torch.nn as nn
import numpy as np

edge_denoiser=nn.Sequential(
    nn.Linear(2073,512),
    nn.ReLU(),#过滤器
    nn.Linear(512,256),
    nn.ReLU(),
    nn.Linear(256,5)
)
###
#原子符号的映射表，方便把原子符号列表挨个映射成one-hot编码的形式
atom_maps={'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'Cl': 5, 'Br': 6, 'F': 7, 'Si': 8};
def get_atom_one_hot(atom_symbols):
    n=len(atom_symbols);
    types=np.zeros((n,10),dtype=np.float32);
    for i,s in enumerate(atom_symbols):
        if s in atom_maps:
            types[i,atom_maps[s]]=1.0;
        else:
            types[i,9]=1.0;#处理未知元素

    return types;
###
def build_init_feature(atom_one_hot,At,fingerprint):
    n=At.shape[0];
    #取上三角所有边的坐标
    i_idx,j_idx=np.triu_indices(n,k=1);
    #创建并初始化特征向量列表
    n_edges=n*(n-1)//2;#边数个2073维向量
    features=np.zeros((n_edges,2073),dtype=np.float32);
    for k in range(n_edges):
        i,j=i_idx[k],j_idx[k];
        features[k,:]=np.concatenate([atom_one_hot[i],
                                      atom_one_hot[j],
                                      At[i,j],
                                      fingerprint]);

    return features;

