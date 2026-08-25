# Day12 消融实验脚本
# 对应计划书：12_ablation.py — 对比有无预训练的差异
# 4组实验：A完整预训练 B无预训练 C仅解码器预训练 D仅编码器预训练

import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置中文字体，避免柱状图中文显示成方块或乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"];
plt.rcParams["axes.unicode_minus"] = False;
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

scripts = r"C:\Users\pc\Desktop\2026DiffMS\scripts";
exec(open(os.path.join(scripts, "04_graph_utils.py"), encoding="utf-8").read());
exec(open(os.path.join(scripts, "05_diffusion.py"), encoding="utf-8").read());
exec(open(os.path.join(scripts, "06_graph_decoder.py"), encoding="utf-8").read());

###############################################
# 一、工具函数
###############################################

def binned_spectrum(mz_list, inten_list, max_mz=1000, bin_size=1.0):
    n_bins = int(max_mz / bin_size); vec = np.zeros(n_bins, dtype=np.float32);
    for mz, inten in zip(mz_list, inten_list):
        if 0 <= mz < max_mz: vec[int(mz/bin_size)] = max(vec[int(mz/bin_size)], float(inten));
    if vec.max() > 0: vec = vec / vec.max();
    return vec;

def build_enc(): return nn.Sequential(nn.Linear(1000,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,2048));
def build_dec(): return nn.Sequential(nn.Linear(2073,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,5));

def get_symbols(atom_types):
    symbols = [];
    for ai in range(atom_types.shape[0]):
        found = False;
        for sym, idx in atom_maps.items():
            if atom_types[ai, idx] == 1.0: symbols.append(sym); found = True; break;
        if not found: symbols.append('C');
    return symbols;

def sample_one(enc, dec, spec_vec, atom_types, n):
    with torch.no_grad():
        condition = enc(torch.tensor(spec_vec, dtype=torch.float32)).numpy();
    At = np.zeros((n,n,5), dtype=np.float32); np.fill_diagonal(At[:,:,0], 1);
    for t in range(499, -1, -1):
        features = build_init_feature(atom_types, At, condition);
        with torch.no_grad():
            preds = torch.argmax(dec(torch.tensor(features)), dim=1).numpy();
        i_idx,j_idx = np.triu_indices(n,k=1);
        At_new = np.zeros_like(At); np.fill_diagonal(At_new[:,:,0], 1);
        for k in range(len(i_idx)):
            At_new[i_idx[k],j_idx[k],preds[k]]=1; At_new[j_idx[k],i_idx[k],preds[k]]=1;
        At = At_new;
    return At;

def evaluate(enc, dec, test_data, name, epochs=20):
    print(f"\n--- 训练组: {name} ---");
    # 用200条数据快速训练
    train_data = test_data[:200];
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-4);
    loss_fn = nn.CrossEntropyLoss(); ab = get_noise_schedule(500);
    for ep in range(epochs):
        total = 0.0;
        for td in train_data:
            spec_t = torch.tensor(td['spec_vec'], dtype=torch.float32);
            cond = enc(spec_t);
            t = sample_timesteps(1,500)[0]; At = q_sample(td['A0'].copy(), t, ab);
            ft = build_init_feature(td['atom_types'], At, cond.detach().numpy());
            n = td['A0'].shape[0]; i_idx,j_idx = np.triu_indices(n,k=1);
            targets = td['A0_labels'][i_idx,j_idx];
            logits = dec(torch.tensor(ft));
            loss = loss_fn(logits, torch.tensor(targets, dtype=torch.long));
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item();
        if ep%5==0: print(f"  Epoch {ep+1}/{epochs}, loss={total/len(train_data):.4f}");

    # 评估
    valid = 0; tan_list = [];
    for td in test_data:
        At_pred = sample_one(enc, dec, td['spec_vec'], td['atom_types'], td['n']);
        symbols = get_symbols(td['atom_types']);
        mol = adj_to_mol(symbols, At_pred);
        if mol is not None:
            valid += 1;
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048);
            ref_bits = "".join([str(int(b)) for b in td['fp']]);
            ref_fp = DataStructs.CreateFromBitString(ref_bits);
            tan_list.append(DataStructs.TanimotoSimilarity(gen_fp, ref_fp));
    val = valid/len(test_data); tan = np.mean(tan_list) if tan_list else 0;
    print(f"  Validity={val:.2%}, Tanimoto={tan:.4f}");
    return val, tan;

###############################################
# 二、加载测试数据（50条）
###############################################

data_dir = r"C:\Users\pc\Desktop\2026DiffMS\data\canopus";
labels_path = os.path.join(data_dir, "labels.tsv");
spec_dir = os.path.join(data_dir, "subformulae", "subformulae_default");

test_data = [];
with open(labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t");
    for row in reader:
        spec = row["spec"]; smiles = row["smiles"];
        jp = os.path.join(spec_dir, spec+".json");
        if not os.path.exists(jp): continue;
        with open(jp) as jf: obj = json.load(jf);
        sv = binned_spectrum(obj["output_tbl"]["mz"], obj["output_tbl"]["ms2_inten"]);
        mol = Chem.MolFromSmiles(smiles);
        if mol is None: continue;
        A0, atom_syms = build_adj(mol);
        test_data.append({'spec_vec':sv, 'A0':A0, 'A0_labels':np.argmax(A0,axis=-1),
                          'atom_types':get_atom_one_hot(atom_syms), 'fp':get_fingerprint(mol), 'n':A0.shape[0]});
        if len(test_data) >= 50: break;

print(f"测试样本: {len(test_data)} 条");

###############################################
# 三、4组消融实验
###############################################

results = {};

# A: 完整预训练（加载Day11模型）
print("A: 完整预训练");
enc_a = build_enc(); enc_a.load_state_dict(torch.load(r"C:\Users\pc\Desktop\2026DiffMS\checkpoints\spec_encoder_day11.pth", map_location="cpu"));
dec_a = build_dec(); dec_a.load_state_dict(torch.load(r"C:\Users\pc\Desktop\2026DiffMS\checkpoints\edge_denoiser_day11.pth", map_location="cpu"));
val_a, tan_a = evaluate(enc_a, dec_a, test_data, "完整预训练", epochs=0);

# B: 无预训练（从头训练）
print("B: 无预训练");
enc_b = build_enc(); dec_b = build_dec();
val_b, tan_b = evaluate(enc_b, dec_b, test_data, "无预训练", epochs=30);

# C: 仅解码器预训练
print("C: 仅解码器预训练");
enc_c = build_enc();
dec_c = build_dec(); dec_c.load_state_dict(torch.load(r"C:\Users\pc\Desktop\2026DiffMS\checkpoints\edge_denoiser_2000.pth", map_location="cpu"));
val_c, tan_c = evaluate(enc_c, dec_c, test_data, "仅解码器预训练", epochs=20);

# D: 仅编码器预训练
print("D: 仅编码器预训练");
enc_d = build_enc(); enc_d.load_state_dict(torch.load(r"C:\Users\pc\Desktop\2026DiffMS\checkpoints\spec_encoder.pth", map_location="cpu"));
dec_d = build_dec();
val_d, tan_d = evaluate(enc_d, dec_d, test_data, "仅编码器预训练", epochs=20);

results = {
    'A-完整预训练': (val_a, tan_a),
    'B-无预训练': (val_b, tan_b),
    'C-仅解码器': (val_c, tan_c),
    'D-仅编码器': (val_d, tan_d),
};

###############################################
# 四、可视化 + 保存
###############################################

os.makedirs("../images/DAY12", exist_ok=True);
os.makedirs("../logs", exist_ok=True);
os.makedirs("../outputs", exist_ok=True);

fig, axes = plt.subplots(1, 2, figsize=(14, 5));
names = list(results.keys());
validities = [results[n][0] for n in names];
tanimotos = [results[n][1] for n in names];

# Validity柱状图
b1 = axes[0].bar(range(len(names)), validities, color=["forestgreen","gray","steelblue","darkorange"]);
axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, rotation=15, fontsize=9);
axes[0].set_ylabel("Validity"); axes[0].set_title("Ablation: Validity Comparison");
for bar, v in zip(b1, validities): axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{v:.0%}", ha="center");
axes[0].grid(axis="y", alpha=0.3);

# Tanimoto柱状图
b2 = axes[1].bar(range(len(names)), tanimotos, color=["forestgreen","gray","steelblue","darkorange"]);
axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=15, fontsize=9);
axes[1].set_ylabel("Tanimoto"); axes[1].set_title("Ablation: Tanimoto Comparison");
for bar, t in zip(b2, tanimotos): axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f"{t:.4f}", ha="center");
axes[1].grid(axis="y", alpha=0.3);

plt.suptitle("DiffMS-Mini Ablation Study", fontsize=14, fontweight="bold");
plt.tight_layout();
plt.savefig("../images/DAY12/ablation_comparison.png", dpi=150, bbox_inches="tight");
print("\n消融对比图已保存");

# 保存CSV
with open("../outputs/ablation_results.csv", "w") as f:
    f.write("Group,Validity,Tanimoto\n");
    for n in names: f.write(f"{n},{results[n][0]:.4f},{results[n][1]:.4f}\n");
print("消融结果已保存到 outputs/ablation_results.csv");

print("\n====== Day12 消融实验完成 ======");



