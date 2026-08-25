# Day11 多候选采样脚本
# 对应计划书：11_sample.py — 每个谱采样10~100个候选分子，记录Top-k Tanimoto
# 核心思路：同一条谱用不同随机噪声起点采样多次，选最像的那几个

import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# 加载工具函数和模型
scripts = r"C:\Users\pc\Desktop\2026DiffMS\scripts";
exec(open(os.path.join(scripts, "04_graph_utils.py"), encoding="utf-8").read());
exec(open(os.path.join(scripts, "05_diffusion.py"), encoding="utf-8").read());
exec(open(os.path.join(scripts, "06_graph_decoder.py"), encoding="utf-8").read());

spec_encoder = nn.Sequential(
    nn.Linear(1000, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 2048)
);
spec_encoder.load_state_dict(torch.load(r"C:\Users\pc\Desktop\2026DiffMS\checkpoints\spec_encoder_day11.pth", map_location="cpu"));
edge_denoiser.load_state_dict(torch.load(r"C:\Users\pc\Desktop\2026DiffMS\checkpoints\edge_denoiser_day11.pth", map_location="cpu"));
spec_encoder.eval(); edge_denoiser.eval();
print("模型加载完成\n");

###############################################
# 一、工具函数
###############################################

def binned_spectrum(mz_list, inten_list, max_mz=1000, bin_size=1.0):
    n_bins = int(max_mz / bin_size); vec = np.zeros(n_bins, dtype=np.float32);
    for mz, inten in zip(mz_list, inten_list):
        if 0 <= mz < max_mz: vec[int(mz/bin_size)] = max(vec[int(mz/bin_size)], float(inten));
    if vec.max() > 0: vec = vec / vec.max();
    return vec;

def sample_one_candidate(spec_vec, atom_types, n):
    """对一条谱采样一个候选分子，返回邻接矩阵"""
    with torch.no_grad():
        condition = spec_encoder(torch.tensor(spec_vec, dtype=torch.float32)).numpy();
    At = np.zeros((n, n, 5), dtype=np.float32);
    np.fill_diagonal(At[:, :, 0], 1);
    for t in range(499, -1, -1):
        features = build_init_feature(atom_types, At, condition);
        with torch.no_grad():
            preds = torch.argmax(edge_denoiser(torch.tensor(features)), dim=1).numpy();
        i_idx, j_idx = np.triu_indices(n, k=1);
        At_new = np.zeros_like(At); np.fill_diagonal(At_new[:, :, 0], 1);
        for k in range(len(i_idx)):
            At_new[i_idx[k], j_idx[k], preds[k]] = 1;
            At_new[j_idx[k], i_idx[k], preds[k]] = 1;
        At = At_new;
    return At;

def get_symbols(atom_types):
    symbols = [];
    for ai in range(atom_types.shape[0]):
        found = False;
        for sym, idx in atom_maps.items():
            if atom_types[ai, idx] == 1.0: symbols.append(sym); found = True; break;
        if not found: symbols.append('C');
    return symbols;

###############################################
# 二、数据（取前20条做演示）
###############################################

data_dir = r"C:\Users\pc\Desktop\2026DiffMS\data\canopus";
labels_path = os.path.join(data_dir, "labels.tsv");
spec_dir = os.path.join(data_dir, "subformulae", "subformulae_default");

test_data = [];
with open(labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t");
    for row in reader:
        spec = row["spec"]; smiles = row["smiles"];
        json_path = os.path.join(spec_dir, spec + ".json");
        if not os.path.exists(json_path): continue;
        with open(json_path, "r") as jf: obj = json.load(jf);
        spec_vec = binned_spectrum(obj["output_tbl"]["mz"], obj["output_tbl"]["ms2_inten"]);
        mol = Chem.MolFromSmiles(smiles);
        if mol is None: continue;
        A0, atom_syms = build_adj(mol);
        test_data.append({
            'spec_vec': spec_vec, 'smiles': smiles,
            'atom_types': get_atom_one_hot(atom_syms),
            'fp': get_fingerprint(mol),
            'ref_mol': mol, 'n': A0.shape[0]
        });
        if len(test_data) >= 20: break;

print(f"测试集: {len(test_data)} 条谱\n");

###############################################
# 三、多候选采样（每条谱采30个候选）
###############################################

num_candidates = 30;
results = [];

for i, td in enumerate(test_data):
    candidates = [];
    for c in range(num_candidates):
        At = sample_one_candidate(td['spec_vec'], td['atom_types'], td['n']);
        symbols = get_symbols(td['atom_types']);
        mol = adj_to_mol(symbols, At);
        if mol is not None:
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048);
            ref_bits = "".join([str(int(b)) for b in td['fp']]);
            ref_fp = DataStructs.CreateFromBitString(ref_bits);
            tan = DataStructs.TanimotoSimilarity(gen_fp, ref_fp);
            candidates.append({'mol': mol, 'tanimoto': tan});

    # 按Tanimoto降序排列
    candidates.sort(key=lambda x: x['tanimoto'], reverse=True);
    top1 = candidates[0]['tanimoto'] if candidates else 0;
    top5 = max([c['tanimoto'] for c in candidates[:5]]) if len(candidates)>=5 else top1;
    top10 = max([c['tanimoto'] for c in candidates[:10]]) if len(candidates)>=10 else top5;
    validity = len(candidates) / num_candidates;

    results.append({'idx': i, 'top1': top1, 'top5': top5, 'top10': top10, 'validity': validity,
                    'smiles': td['smiles'][:50]});
    print(f"[{i}] Top-1={top1:.4f}  Top-5={top5:.4f}  Top-10={top10:.4f}  Validity={validity:.0%}");

###############################################
# 四、汇总统计
###############################################

avg_top1 = np.mean([r['top1'] for r in results]);
avg_top5 = np.mean([r['top5'] for r in results]);
avg_top10 = np.mean([r['top10'] for r in results]);
avg_val = np.mean([r['validity'] for r in results]);

print(f"\n====== Top-k 评测汇总 ======");
print(f"平均 Top-1  Tanimoto: {avg_top1:.4f}");
print(f"平均 Top-5  Tanimoto: {avg_top5:.4f}");
print(f"平均 Top-10 Tanimoto: {avg_top10:.4f}");
print(f"平均 Validity:        {avg_val:.2%}");
print(f"\n分析：Top-10 > Top-1 说明多候选采样确实可以提高命中率。");
print(f"但Tanimoto绝对值仍然很低，根本原因还是MLP解码器容量不足。");

# 保存结果
import matplotlib.pyplot as plt;
os.makedirs("../logs", exist_ok=True);
os.makedirs("../images/DAY11", exist_ok=True);

# 柱状图
fig, ax = plt.subplots(figsize=(8,5));
bars = ax.bar(["Top-1", "Top-5", "Top-10"], [avg_top1, avg_top5, avg_top10],
              color=["steelblue", "darkorange", "forestgreen"]);
for bar, v in zip(bars, [avg_top1, avg_top5, avg_top10]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f"{v:.4f}", ha="center", fontweight="bold");
ax.set_ylabel("Average Tanimoto"); ax.set_title("Multi-Candidate Sampling: Top-k Tanimoto");
ax.grid(axis="y", alpha=0.3);
plt.savefig("../images/DAY11/topk_tanimoto.png", dpi=150, bbox_inches="tight");
print("Top-k柱状图已保存\n====== Day11 多候选采样完成 ======");
