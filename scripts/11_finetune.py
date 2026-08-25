# Day11 端到端微调训练脚本
# 对应计划书：11_finetune.py — 在小数据集上训练 spectrum→graph
# 核心改进（相比Day10）：更多轮次（80轮），按计划书要求完成

import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# 加载前期工具函数
scripts = r"C:\Users\pc\Desktop\2026DiffMS\scripts";
exec(open(os.path.join(scripts, "04_graph_utils.py"), encoding="utf-8").read());
exec(open(os.path.join(scripts, "05_diffusion.py"), encoding="utf-8").read());
exec(open(os.path.join(scripts, "06_graph_decoder.py"), encoding="utf-8").read());

###############################################
# 一、加载模型
###############################################

spec_encoder = nn.Sequential(
    nn.Linear(1000, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 2048)
);
enc_path = r"C:\Users\pc\Desktop\2026DiffMS\checkpoints\spec_encoder.pth";
if os.path.exists(enc_path):
    spec_encoder.load_state_dict(torch.load(enc_path, map_location="cpu"));
    print("谱编码器已加载");

dec_path = r"C:\Users\pc\Desktop\2026DiffMS\checkpoints\edge_denoiser_2000.pth";
if os.path.exists(dec_path):
    edge_denoiser.load_state_dict(torch.load(dec_path, map_location="cpu"));
    print("图解码器已加载");

print("准备端到端微调（80轮）\n");

###############################################
# 二、谱图向量化
###############################################

def binned_spectrum(mz_list, inten_list, max_mz=1000, bin_size=1.0):
    n_bins = int(max_mz / bin_size);
    vec = np.zeros(n_bins, dtype=np.float32);
    for mz, inten in zip(mz_list, inten_list):
        if 0 <= mz < max_mz:
            idx = int(mz / bin_size);
            vec[idx] = max(vec[idx], float(inten));
    if vec.max() > 0: vec = vec / vec.max();
    return vec;

###############################################
# 三、加载数据
###############################################

data_dir = r"C:\Users\pc\Desktop\2026DiffMS\data\canopus";
labels_path = os.path.join(data_dir, "labels.tsv");
spec_dir = os.path.join(data_dir, "subformulae", "subformulae_default");

all_data = [];
with open(labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t");
    for row in reader:
        spec = row["spec"]; smiles = row["smiles"];
        json_path = os.path.join(spec_dir, spec + ".json");
        if not os.path.exists(json_path): continue;
        with open(json_path, "r") as jf:
            obj = json.load(jf);
        spec_vec = binned_spectrum(obj["output_tbl"]["mz"], obj["output_tbl"]["ms2_inten"]);
        mol = Chem.MolFromSmiles(smiles);
        if mol is None: continue;
        A0, atom_symbols = build_adj(mol);
        all_data.append({
            'smiles': smiles, 'spec_vec': spec_vec,
            'A0': A0, 'A0_labels': np.argmax(A0, axis=-1),
            'atom_types': get_atom_one_hot(atom_symbols), 'fp': get_fingerprint(mol)
        });
        if len(all_data) >= 500: break;

print(f"加载 {len(all_data)} 条样本\n");

###############################################
# 四、端到端训练（80轮）
###############################################

optimizer = torch.optim.Adam(
    list(spec_encoder.parameters()) + list(edge_denoiser.parameters()), lr=1e-4
);
loss_fn = nn.CrossEntropyLoss();
alpha_bars = get_noise_schedule(500);
num_epochs = 80;
loss_history = [];

print("====== 开始训练 ======");
for epoch in range(num_epochs):
    total_loss = 0.0;
    for data in all_data:
        spec_t = torch.tensor(data['spec_vec'], dtype=torch.float32);
        condition = spec_encoder(spec_t);
        t = sample_timesteps(1, 500)[0];
        At = q_sample(data['A0'].copy(), t, alpha_bars);
        features = build_init_feature(data['atom_types'], At, condition.detach().numpy());
        n = data['A0'].shape[0];
        i_idx, j_idx = np.triu_indices(n, k=1);
        targets = data['A0_labels'][i_idx, j_idx];
        logits = edge_denoiser(torch.tensor(features));
        loss = loss_fn(logits, torch.tensor(targets, dtype=torch.long));
        optimizer.zero_grad(); loss.backward(); optimizer.step();
        total_loss += loss.item();
    avg_loss = total_loss / len(all_data);
    loss_history.append(avg_loss);
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1:2d}/{num_epochs}, loss = {avg_loss:.4f}");

###############################################
# 五、保存
###############################################

os.makedirs("../images/DAY11", exist_ok=True);
plt.figure(figsize=(8,4));
plt.plot(loss_history); plt.xlabel("Epoch"); plt.ylabel("Loss");
plt.title("Day11 End-to-End Finetuning (80 epochs)");
plt.grid(True); plt.savefig("../images/DAY11/loss_curve.png", dpi=150, bbox_inches="tight");
print("\nLoss曲线已保存");

torch.save(spec_encoder.state_dict(), "../checkpoints/spec_encoder_day11.pth");
torch.save(edge_denoiser.state_dict(), "../checkpoints/edge_denoiser_day11.pth");
print("模型已保存");

os.makedirs("../logs", exist_ok=True);
with open("../logs/day11_finetune_log.txt", "w") as f:
    for i, loss in enumerate(loss_history):
        f.write(f"Epoch {i+1}: loss={loss:.4f}\n");

print("====== Day11 端到端微调完成 ======");
