# ========== 定义 get_fingerprint 函数（你第2天写的） ==========
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def get_fingerprint(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)

# ========== 读取分子，生成指纹 ==========
smiles = "CC(=O)O"
mol = Chem.MolFromSmiles(smiles)

fp_array = get_fingerprint(mol)

# ========== 画竖线图 ==========
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# 设置中文字体
try:
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=14)
except:
    font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=14)

fig, ax = plt.subplots(figsize=(16, 4))
ones_positions = np.where(fp_array == 1)[0]
ax.vlines(ones_positions, 0, 1, color='blue', linewidth=0.8)

ax.set_title("乙酸（Acetic Acid）的 Morgan 指纹", fontproperties=font, fontsize=16)
ax.set_xlabel("2048 位", fontproperties=font, fontsize=12)
ax.set_ylabel("1 = 有此片段", fontproperties=font, fontsize=12)

ax.set_xlim(-10, 2058)
ax.set_ylim(0, 1.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(['0', '1'])

os.makedirs('images', exist_ok=True)
save_path = 'images/乙酸_摩根指纹_竖线图.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图片已保存到: {save_path}")

plt.show()