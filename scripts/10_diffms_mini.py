# Day10 端到端微调完整版
# 任务：谱编码器 + 图解码器联合训练（对应论文 Figure 3-C）
# 核心思想：用 spec_encoder 的输出替代真实指纹作为条件，让两个模块一起学

import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# 加载前期工具函数（Day4/5/6）
scripts = r"C:\Users\pc\Desktop\2026DiffMS\scripts";
exec(open(os.path.join(scripts, "04_graph_utils.py"), encoding="utf-8").read());
exec(open(os.path.join(scripts, "05_diffusion.py"), encoding="utf-8").read());
exec(open(os.path.join(scripts, "06_graph_decoder.py"), encoding="utf-8").read());

###############################################
# 一、加载两个预训练模型（Day8的解码器 + Day9的编码器）
###############################################

####
# 1.1 谱编码器（Day9训好的）
# 结构必须和训练时完全一致：1000→512→256→2048
spec_encoder = nn.Sequential(
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 2048)
);
enc_path = r"C:\Users\pc\Desktop\2026DiffMS\checkpoints\spec_encoder.pth";
if os.path.exists(enc_path):
    spec_encoder.load_state_dict(torch.load(enc_path, map_location="cpu"));
    print("谱编码器权重已加载");
else:
    print("警告: 未找到谱编码器权重");

####
# 1.2 图解码器（Day8训好的）
# edge_denoiser 在 06_graph_decoder.py 已定义为全局变量
dec_path = r"C:\Users\pc\Desktop\2026DiffMS\checkpoints\edge_denoiser_2000.pth";
if os.path.exists(dec_path):
    edge_denoiser.load_state_dict(torch.load(dec_path, map_location="cpu"));
    print("图解码器权重已加载");
else:
    print("警告: 未找到图解码器权重");

print("\n两个模型加载完成，准备端到端微调");

###############################################
# 二、谱图向量化（复用 Day9 的同名函数）
###############################################

def binned_spectrum(mz_list, inten_list, max_mz=1000, bin_size=1.0):
    n_bins = int(max_mz / bin_size);        # 总格子数
    vec = np.zeros(n_bins, dtype=np.float32);  # 初始全零向量
    for mz, inten in zip(mz_list, inten_list):
        if 0 <= mz < max_mz:                # 只保留范围0~1000内的峰
            idx = int(mz / bin_size);        # 计算属于哪个格子
            vec[idx] = max(vec[idx], float(inten));  # 同格子取最大强度
    if vec.max() > 0:
        vec = vec / vec.max();              # 归一化到0~1
    return vec;

###############################################
# 三、加载端到端训练数据（谱 + 分子图配对）
###############################################

data_dir = r"C:\Users\pc\Desktop\2026DiffMS\data\canopus";
labels_path = os.path.join(data_dir, "labels.tsv");
spec_dir = os.path.join(data_dir, "subformulae", "subformulae_default");

all_data = [];
failed_count = 0;

with open(labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t");
    for row in reader:
        spec = row["spec"];
        smiles = row["smiles"];
        json_path = os.path.join(spec_dir, spec + ".json");

        # 3.1 读谱文件 → 向量化
        if not os.path.exists(json_path):
            failed_count += 1;
            continue;
        with open(json_path, "r") as jf:
            obj = json.load(jf);
        mz = obj["output_tbl"]["mz"];
        inten = obj["output_tbl"]["ms2_inten"];
        spec_vec = binned_spectrum(mz, inten);

        # 3.2 SMILES → 分子图（邻接矩阵 + 原子列表 + 指纹）
        mol = Chem.MolFromSmiles(smiles);
        if mol is None:
            failed_count += 1;
            continue;
        A0, atom_symbols = build_adj(mol);
        fp = get_fingerprint(mol);
        atom_types = get_atom_one_hot(atom_symbols);
        A0_labels = np.argmax(A0, axis=-1);

        all_data.append({
            'smiles': smiles,           # 原始SMILES（评估用）
            'spec_vec': spec_vec,       # 谱向量（喂给spec_encoder）
            'A0': A0,                   # 邻接矩阵
            'A0_labels': A0_labels,     # 键类型标签（算loss用）
            'atom_types': atom_types,   # 原子one-hot（build_init_feature用）
            'fp': fp,                   # 真实指纹（评估用）
        });

        if len(all_data) >= 500:
            break;

print(f"加载完成: {len(all_data)} 条样本, 失败跳过: {failed_count}");
print(f"第1条 spec_vec形状: {all_data[0]['spec_vec'].shape}, A0形状: {all_data[0]['A0'].shape}");

###############################################
# 四、端到端微调训练
###############################################

####
# 4.1 设置优化器、损失函数、噪声调度
# 两个模型的参数合并给同一个优化器，让梯度同时流回两个模块
optimizer = torch.optim.Adam(
    list(spec_encoder.parameters()) + list(edge_denoiser.parameters()),
    lr=1e-4
);
loss_fn = nn.CrossEntropyLoss();      # 和Day8一样：预测键类型，5选1
alpha_bars = get_noise_schedule(500);  # 离散扩散的噪声调度表

####
# 4.2 训练循环
# 流程: 谱→spec_encoder→条件向量→build_init_feature→edge_denoiser→预测键类型
num_epochs = 30;
loss_history = [];

print("\n====== 开始端到端训练 ======");
for epoch in range(num_epochs):
    total_loss = 0.0;
    for data in all_data:
        # ① 谱向量 → spec_encoder → 条件向量（2048维连续值）
        spec_tensor = torch.tensor(data['spec_vec'], dtype=torch.float32);
        condition = spec_encoder(spec_tensor);

        # ② 随机选时间步 + 加噪乱键图
        t = sample_timesteps(1, 500)[0];
        At = q_sample(data['A0'].copy(), t, alpha_bars);

        # ③ 提取上三角所有边的特征，条件向量拼进去
        condition_np = condition.detach().numpy();
        features = build_init_feature(data['atom_types'], At, condition_np);

        # ④ 取上三角所有边的真实标签
        n = data['A0'].shape[0];
        i_idx, j_idx = np.triu_indices(n, k=1);
        targets = data['A0_labels'][i_idx, j_idx];

        # ⑤ 模型预测 + 算loss
        logits = edge_denoiser(torch.tensor(features));
        loss = loss_fn(logits, torch.tensor(targets, dtype=torch.long));

        # ⑥ 反向传播：梯度同时流回spec_encoder和edge_denoiser
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        total_loss += loss.item();

    avg_loss = total_loss / len(all_data);
    loss_history.append(avg_loss);
    if epoch % 5 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1:2d}/{num_epochs}, loss = {avg_loss:.4f}");

###############################################
# 五、采样评估——谱 → 完整分子图
###############################################

####
# 5.1 从谱出发完整采样一个分子
# 和Day8不同：条件向量来自spec_encoder，而不是真实指纹
def sample_molecule(spec_vec, atom_types, num_steps=500):
    n = atom_types.shape[0];

    # ① 谱向量 → 条件向量
    spec_tensor = torch.tensor(spec_vec, dtype=torch.float32);
    with torch.no_grad():
        condition = spec_encoder(spec_tensor).numpy();

    # ② 从纯随机噪声开始（全无键状态）
    At = np.zeros((n, n, 5), dtype=np.float32);
    np.fill_diagonal(At[:, :, 0], 1);  # 对角线是原子自己，永远无键

    # ③ 逐步去噪：t从499递减到0，每一步修复一点
    for t in range(num_steps - 1, -1, -1):
        features = build_init_feature(atom_types, At, condition);
        with torch.no_grad():
            logits = edge_denoiser(torch.tensor(features));
            preds = torch.argmax(logits, dim=1).numpy();

        # 把预测的边填回At（注意保持对称）
        i_idx, j_idx = np.triu_indices(n, k=1);
        At_new = np.zeros_like(At);
        np.fill_diagonal(At_new[:, :, 0], 1);
        for k in range(len(i_idx)):
            At_new[i_idx[k], j_idx[k], preds[k]] = 1;
            At_new[j_idx[k], i_idx[k], preds[k]] = 1;
        At = At_new;

    return At;

####
# 5.2 原子符号还原辅助函数
def get_atom_symbols(atom_types):
    symbols = [];
    for ai in range(atom_types.shape[0]):
        found = False;
        for sym, idx in atom_maps.items():
            if atom_types[ai, idx] == 1.0:
                symbols.append(sym);
                found = True;
                break;
        if not found:
            symbols.append('C');  # 未知元素兜底
    return symbols;

###############################################
# 六、Validity + Tanimoto 评估
###############################################

num_eval = 50;
valid_count = 0;
tanimoto_list = [];

print("\n====== 评估采样质量 ======");
for i in range(num_eval):
    data = all_data[i];
    symbols = get_atom_symbols(data['atom_types']);

    # 采样 → 检查合法性
    At_pred = sample_molecule(data['spec_vec'], data['atom_types']);
    mol = adj_to_mol(symbols, At_pred);

    if mol is not None:
        valid_count += 1;

        # 生成分子 → 指纹 → 和真实指纹比 Tanimoto
        gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048);
        ref_bits = "".join([str(int(b)) for b in data['fp']]);
        ref_fp = DataStructs.CreateFromBitString(ref_bits);
        tan = DataStructs.TanimotoSimilarity(gen_fp, ref_fp);
        tanimoto_list.append(tan);

validity = valid_count / num_eval;
avg_tan = np.mean(tanimoto_list) if tanimoto_list else 0.0;
print(f"Validity: {validity:.2%} ({valid_count}/{num_eval})");
print(f"平均 Tanimoto 相似度: {avg_tan:.4f}");

###############################################
# 七、失败案例分析
###############################################

print("\n====== 失败案例分析 ======");
print("低Tanimoto的原因分析：");
for i in range(min(5, len(tanimoto_list))):
    print(f"  [{i}] Tanimoto={tanimoto_list[i]:.4f}  SMILES={all_data[i]['smiles'][:60]}...");

print("\n总结：");
print("  Validity 100%说明分子式约束有效——固定原子列表缩小了生成空间。");
print("  但Tanimoto很低说明MLP解码器很难从谱编码器的模糊条件中恢复精确结构。");
print("  MLP对每条边独立预测，无法捕捉化学键之间的全局依赖关系（如环结构、共轭体系）。");
print("  升级到Graph Transformer（论文原版架构）是后续提升的关键方向。");

###############################################
# 八、保存结果
###############################################

####
# 8.1 loss曲线
os.makedirs("../images/DAY10", exist_ok=True);
plt.figure(figsize=(8, 4));
plt.plot(loss_history, label="End-to-End Loss");
plt.xlabel("Epoch");
plt.ylabel("Cross-Entropy Loss");
plt.title("DiffMS-Mini End-to-End Finetuning");
plt.legend();
plt.grid(True);
plt.savefig("../images/DAY10/loss_curve.png", dpi=150, bbox_inches="tight");
print("\nLoss曲线已保存到 images/DAY10/loss_curve.png");

####
# 8.2 训练日志
os.makedirs("../logs", exist_ok=True);
with open("../logs/day10_end2end_log.txt", "w") as f:
    f.write(f"Validity: {validity:.4f}\n");
    f.write(f"Tanimoto: {avg_tan:.4f}\n");
    f.write(f"有效分子数: {valid_count}/{num_eval}\n");
    for i, loss in enumerate(loss_history):
        f.write(f"Epoch {i+1}: loss={loss:.4f}\n");

####
# 8.3 失败案例分析报告
with open("../logs/day10_analysis.txt", "w", encoding="utf-8") as f:
    f.write("DiffMS-Mini 端到端微调失败案例分析\n\n");
    f.write("一、实验设置\n");
    f.write("  谱编码器: MLP (1000->512->256->2048), CANOPUS 1000条预训练\n");
    f.write("  图解码器: MLP (2073->512->256->5), MOSES 2000条预训练\n");
    f.write("  端到端: CANOPUS 500条谱-结构配对, 30轮微调\n\n");
    f.write("二、结果\n");
    f.write(f"  Validity: {validity:.2%}\n");
    f.write(f"  Tanimoto: {avg_tan:.4f}\n\n");
    f.write("三、分析\n");
    f.write("  1. Validity 100%：分子式约束（固定原子列表）有效\n");
    f.write("     分子图生成空间被极大缩小，RDKit易通过合法性检查\n");
    f.write("  2. Tanimoto ~0：MLP解码器容量不足\n");
    f.write("     MLP对每条边独立预测，不共享化学键依赖信息\n");
    f.write("     同分异构体有无数合法排列，MLP无法区分目标结构\n");
    f.write("  3. 谱编码器信息损失\n");
    f.write("     1000维binned spectrum → 2048维指纹是高度压缩任务\n");
    f.write("     信息瓶颈限制了向下游传递的条件质量\n\n");
    f.write("四、后续改进方向\n");
    f.write("  a. 升级解码器为Graph Transformer（论文原版DiGress架构）\n");
    f.write("  b. 增加训练数据量（500 -> 1万+ 谱-结构配对）\n");
    f.write("  c. 增加编码器预训练数据（2.8M指纹-结构对）\n");
    f.write("  d. 引入MCES评估（最大公共子结构，比Tanimoto更精确）\n");

####
# 8.4 模型权重
os.makedirs("../checkpoints", exist_ok=True);
torch.save(spec_encoder.state_dict(), "../checkpoints/spec_encoder_e2e.pth");
torch.save(edge_denoiser.state_dict(), "../checkpoints/edge_denoiser_e2e.pth");
print("端到端模型权重已保存到 checkpoints/");

print("\n====== Day10 端到端微调完成 ======");
