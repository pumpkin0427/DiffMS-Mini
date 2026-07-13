from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SanitizeMol
import numpy as np

smiles = "CC(=O)O"
mol = Chem.MolFromSmiles(smiles)


def get_atom_list(mol):
    atom_list = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            atom_list.append(atom.GetSymbol())
    return atom_list
###
def build_adj(mol):
    # 1. 获取重原子列表
    heavy_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            heavy_atoms.append(atom)
    n = len(heavy_atoms)

    # 2. 创建索引映射
    idx_map = {}
    i = 0
    for atom in heavy_atoms:
        idx_map[atom.GetIdx()] = i
        i = i + 1

    # 3. 初始化邻接矩阵
    A = np.zeros((n, n, 5), dtype=np.float32)

    # 4. 遍历所有键，填充矩阵
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if begin_idx in idx_map and end_idx in idx_map:
            i = idx_map[begin_idx]
            j = idx_map[end_idx]

            bond_type = bond.GetBondType()
            if bond_type == Chem.rdchem.BondType.SINGLE:
                k = 1
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                k = 2
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                k = 3
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                k = 4
            else:
                k = 0

            A[i, j, k] = 1
            A[j, i, k] = 1
    # 返回邻接矩阵和原子符号列表
    atom_symbols = []
    for atom in heavy_atoms:
        atom_symbols.append(atom.GetSymbol())
    return A, atom_symbols;
###
def get_fingerprint(mol):
    # 1. 生成 Morgan 指纹（半径2，长度2048）
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    # 2. 转换成 NumPy 数组（0/1 向量）
    fp_array = np.array(fp)
    # 3. 返回
    return fp_array
###
def adj_to_mol(atom_symbols,adj):
    #1.创建空分子对象；
    mol=Chem.RWMol();
    #2.添加原子：
    for symbol in atom_symbols:
        atom=Chem.Atom(symbol)
        mol.AddAtom(atom);
    #3.遍历邻接矩阵，添加键
    bond_map = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.AROMATIC,
    }
    n=len(atom_symbols);
    for i in range(n):
        for j in range(i+1,n):
            bond_type_idx=np.argmax(adj[i,j,:]);
            if bond_type_idx in bond_map:
                mol.AddBond(i,j,bond_map[bond_type_idx]);
    try:
        Chem.SanitizeMol(mol);
        return mol.GetMol();
    except Exception as e:
        print(f"分子不合法:{e}");
        return None;