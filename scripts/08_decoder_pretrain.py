import os
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))

# 加载已有模块
exec(open(os.path.join(script_dir, '04_graph_utils.py'), encoding='utf-8').read())
exec(open(os.path.join(script_dir, '05_diffusion.py'), encoding='utf-8').read())
exec(open(os.path.join(script_dir, '06_graph_decoder.py'), encoding='utf-8').read())

# ===== 第一步：从文件加载 SMILES =====
def load_smiles(path):
    valid = []
    with open(path, 'r') as f:
        for line in f:
            smi = line.strip()
            if smi and Chem.MolFromSmiles(smi):
                valid.append(smi)
    print(f"加载 {len(valid)} 个有效 SMILES")
    return valid

smiles_list = load_smiles(os.path.join(project_dir, 'data', 'smiles_2k.txt'))

# ===== 第二步：预处理所有分子 =====
all_data = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    A0, atom_symbols = build_adj(mol)
    fp = get_fingerprint(mol)
    atom_types = get_atom_one_hot(atom_symbols)
    A0_labels = np.argmax(A0, axis=-1)
    all_data.append({
        'A0': A0,
        'A0_labels': A0_labels,
        'atom_types': atom_types,
        'fp': fp,
    })
print(f"预处理完成，共 {len(all_data)} 个分子")