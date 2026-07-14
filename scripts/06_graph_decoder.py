import torch
import torch.nn as nn#神经网络模块
import numpy as np

#创建MLP模型:输入2073维→输出5维(5种键类型)
edge_denoiser=nn.Sequential(
    nn.Linear(2073,512),#第一层:压缩信息
    nn.ReLU(),
    nn.Linear(512,256),#第二层:进一步提炼
    nn.ReLU(),
    nn.Linear(256,5),#输出层:5种键类型的分数
);

####
#原子类型映射表
atom_type_map={'C':0,'N':1,'O':2,'S':3,'P':4,'Cl':5,'Br':6,'F':7,'Si':8};

def get_atom_types(atom_symbols):
    #把原子符号列表转成one-hot矩阵 (n,10)
    n=len(atom_symbols);
    types=np.zeros((n,10),dtype=np.float32);
    for i,s in enumerate(atom_symbols):
        if s in atom_type_map:
            types[i,atom_type_map[s]]=1.0;
        else:
            types[i,9]=1.0;#未知类型
    return types;

####
def build_edge_features(atom_types,At,fingerprint):
    #对上三角每条边构建MLP输入特征
    n=At.shape[0];
    i_idx,j_idx=np.triu_indices(n,k=1);
    features=[];
    for k in range(len(i_idx)):
        i=i_idx[k];
        j=j_idx[k];
        #拼接: 原子i + 原子j + 噪声边 + 指纹
        feat=np.concatenate([
            atom_types[i],
            atom_types[j],
            At[i,j],
            fingerprint
        ]);
        features.append(feat);
    return np.array(features,dtype=np.float32);