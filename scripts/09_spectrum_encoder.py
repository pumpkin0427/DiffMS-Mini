# 任务：把质谱峰数据 → 固定长度向量 → MLP预测分子指纹
# 对应论文 Figure 3-A：编码器预训练——"看谱猜结构"

import os
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

# 加载前期写好的工具函数（Day4/5/6 的代码）
script_dir = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(script_dir, '04_graph_utils.py'), encoding='utf-8').read());

###############################################
# 一、数据加载
###############################################

# 路径配置
data_dir = r"C:\Users\pc\Desktop\2026DiffMS\data\canopus"
labels_path = os.path.join(data_dir, "labels.tsv")
spec_dir = os.path.join(data_dir, "subformulae", "subformulae_default")

####
# 1.1 从 labels.tsv 读取样本元数据
# 每一行包含：spec（谱编号）、smiles（分子结构）、formula（分子式）等
# 我们只需要 spec 和 smiles：spec用来找JSON谱文件，smiles用来算指纹
samples = []
with open(labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        spec = row["spec"]
        smiles = row["smiles"]
        json_path = os.path.join(spec_dir, spec + ".json")
        # 只保留谱文件存在且分子结构可解析的样本
        if os.path.exists(json_path) and Chem.MolFromSmiles(smiles) is not None:
            samples.append({"spec": spec, "smiles": smiles, "json_path": json_path})
        if len(samples) >= 1000:  # 先用1000条做小规模实验
            break
print(f"加载样本数: {len(samples)}");

###############################################
# 二、谱图向量化（Binned Spectrum）
###############################################

####
# 2.1 把不定长的峰列表 → 固定长度的向量
# 原理：m/z 范围 0~1000，每1单位一格，共1000格
# 每个峰根据 m/z 值落入对应格子，格子里存该位置最强峰的 intensity
# 最后归一化，让所有谱在同一尺度上
def binned_spectrum(mz_list, inten_list, max_mz=1000, bin_size=1.0):
    n_bins = int(max_mz / bin_size);        # 总格子数
    vec = np.zeros(n_bins, dtype=np.float32);  # 初始全零向量

    for mz, inten in zip(mz_list, inten_list):
        if 0 <= mz < max_mz:                # 只保留范围内的峰
            idx = int(mz / bin_size);        # 计算属于哪个格子
            vec[idx] = max(vec[idx], float(inten));  # 同格子取最大值

    # 归一化：让每个谱的最大强度变成1.0
    if vec.max() > 0:
        vec = vec / vec.max();
    return vec;

###############################################
# 三、准备训练数据（X：谱向量，y：指纹）
###############################################

####
# 3.1 遍历所有样本，生成 X 和 y
# X：谱向量（1000维）——模型输入
# y：Morgan指纹（2048位，0/1）——模型要预测的目标
X_list = [];   # 存每个分子的谱向量
y_list = [];   # 存每个分子的指纹
for i, s in enumerate(samples):
    # 3.1a 读JSON谱文件 → 取 m/z 和 intensity → 向量化
    with open(s["json_path"], "r") as f:
        obj = json.load(f);
    mz = obj["output_tbl"]["mz"];
    inten = obj["output_tbl"]["ms2_inten"];
    spec_vec = binned_spectrum(mz, inten);

    # 3.1b 从SMILES生成分子指纹
    mol = Chem.MolFromSmiles(s["smiles"]);
    fp = get_fingerprint(mol);   # 来自04_graph_utils.py

    X_list.append(spec_vec);
    y_list.append(fp);

    if i % 200 == 0:
        print(f"  处理中... {i}/{len(samples)}");
print(f"数据准备完成: X={len(X_list)}条, y={len(y_list)}条");
print(f"X单条形状: {X_list[0].shape}, y单条形状: {y_list[0].shape}");

####
# 3.2 转成PyTorch矩阵，划分训练集/测试集（8:2）
X = np.stack(X_list, axis=0);     # (1000, 1000)
y = np.stack(y_list, axis=0);     # (1000, 2048)

X_tensor = torch.tensor(X, dtype=torch.float32);
y_tensor = torch.tensor(y, dtype=torch.float32);

X_train, X_test = X_tensor[:800], X_tensor[800:];
y_train, y_test = y_tensor[:800], y_tensor[800:];
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}");

# 用DataLoader做mini-batch训练（每批32条）
train_dataset = TensorDataset(X_train, y_train);
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True);

###############################################
# 四、搭建模型
###############################################

####
# 4.1 谱编码器 SpectumEncoder
# 结构：3层全连接MLP
# 输入：1000维谱向量
# 输出：2048维指纹预测（每个数字代表指纹对应位的"得分"）
# 注意：最后一层没有Sigmoid，因为BCEWithLogitsLoss自己会处理
spec_encoder = nn.Sequential(
    nn.Linear(1000, 512),   # 第1层：1000个谱bin → 512维隐藏特征
    nn.ReLU(),              # 激活函数，过滤负值
    nn.Linear(512, 256),    # 第2层：512 → 256
    nn.ReLU(),
    nn.Linear(256, 2048)    # 第3层：256 → 2048位指纹预测
);

####
# 4.2 损失函数和优化器
# BCEWithLogitsLoss：多标签二分类损失，适合2048位独立0/1判断
# Adam：自适应学习率优化器，lr=1e-4是常用的初始学习率
loss_fn = nn.BCEWithLogitsLoss();
optimizer = torch.optim.Adam(spec_encoder.parameters(), lr=1e-4);
print(f"模型参数量: {sum(p.numel() for p in spec_encoder.parameters())}");

###############################################
# 五、训练
###############################################

num_epochs = 50;
loss_history = [];

print("\n====== 开始训练 ======");
for epoch in range(num_epochs):
    total_loss = 0.0;
    for batch_x, batch_y in train_loader:
        # 5.1 前向传播：谱向量 → 预测指纹
        pred = spec_encoder(batch_x);
        # 5.2 算损失：预测指纹 vs 真实指纹
        loss = loss_fn(pred, batch_y);
        # 5.3 反向传播 + 参数更新（和Day6/7一样的套路）
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        total_loss += loss.item();

    avg_loss = total_loss / len(train_loader);
    loss_history.append(avg_loss);
    print(f"Epoch {epoch+1:2d}/{num_epochs}, loss = {avg_loss:.4f}");

###############################################
# 六、测试集评估
###############################################

####
# 6.1 用Tanimoto相似度衡量预测质量
# Tanimoto = 预测和真实都为1的位数 / 至少一个为1的位数
# 范围0~1，越大越好；瞎猜的基线约0.01~0.03
with torch.no_grad():
    pred_logits = spec_encoder(X_test);            # 原始得分
    pred_probs = torch.sigmoid(pred_logits);        # 转成0~1概率
    pred_bits = (pred_probs > 0.5).float();         # >0.5判为1，否则为0

    intersection = (pred_bits * y_test).sum(dim=1);       # 两个都是1的位数
    union = ((pred_bits + y_test) > 0).float().sum(dim=1); # 至少一个是1的位数
    tanimoto = (intersection / (union + 1e-8)).mean().item();

print(f"\n测试集平均 Tanimoto 相似度: {tanimoto:.4f}");
print(f"（随机基线约0.02，你的模型是它的 {tanimoto/0.02:.0f}x 倍）");

###############################################
# 七、保存结果
###############################################

# 7.1 画loss曲线
os.makedirs("../images/DAY9", exist_ok=True);
plt.figure(figsize=(8, 4));
plt.plot(loss_history, label="Training Loss");
plt.xlabel("Epoch");
plt.ylabel("Loss");
plt.title("Spectrum Encoder Training Loss (BCEWithLogits)");
plt.legend();
plt.grid(True);
plt.savefig("../images/DAY9/loss_curve.png", dpi=150, bbox_inches="tight");
print("\nLoss曲线已保存到 images/DAY9/loss_curve.png");

# 7.2 保存训练日志
os.makedirs("../logs", exist_ok=True);
with open("../logs/day9_encoder_log.txt", "w") as f:
    f.write(f"Test Tanimoto: {tanimoto:.4f}\n");
    for i, loss in enumerate(loss_history):
        f.write(f"Epoch {i+1}: loss={loss:.4f}\n");

# 7.3 保存模型权重
torch.save(spec_encoder.state_dict(), "../checkpoints/spec_encoder.pth");
print("模型权重已保存到 checkpoints/spec_encoder.pth");

print("\nDay9 谱编码器训练完成");
