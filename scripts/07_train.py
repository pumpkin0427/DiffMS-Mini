import os
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

#获取当前脚本所在目录
script_dir=os.path.dirname(os.path.abspath(__file__));
#从同一目录加载其他脚本
exec(open(os.path.join(script_dir,'04_graph_utils.py'),encoding='utf-8').read());
exec(open(os.path.join(script_dir,'05_diffusion.py'),encoding='utf-8').read());
exec(open(os.path.join(script_dir,'06_graph_decoder.py'),encoding='utf-8').read());

####
#1.准备5个分子的训练数据
smiles_list=['CC(=O)O','CCO','CCN','c1ccccc1','CCC'];

all_data=[];
for smiles in smiles_list:
    mol=Chem.MolFromSmiles(smiles);
    A0,atom_symbols=build_adj(mol);#邻接矩阵 (n,n,5)
    fp=get_fingerprint(mol);#指纹 (2048,)
    atom_types=get_atom_types(atom_symbols);#原子类型 (n,10)
    A0_labels=np.argmax(A0,axis=-1);#真实键类型 (n,n)
    all_data.append({
        'A0':A0,
        'A0_labels':A0_labels,
        'atom_types':atom_types,
        'fp':fp,
    });
print(f"已准备{len(all_data)}个分子的数据");

####
#2.设置优化器和损失函数
optimizer=torch.optim.Adam(edge_denoiser.parameters(),lr=1e-4);
loss_fn=nn.CrossEntropyLoss();
alpha_bars=get_noise_schedule(500);

####
#3.训练循环
num_epochs=100;#CPU跑得慢，先100轮看趋势
loss_history=[];#记录每一轮的loss

for epoch in range(num_epochs):
    total_loss=0.0;
    for data in all_data:#遍历每个分子
        #3.1 随机选时间步
        t=sample_timesteps(1,500)[0];
        #3.2 加噪: A0→At
        At=q_sample(data['A0'].copy(),t,alpha_bars);
        #3.3 提取上三角所有边的特征
        features=build_edge_features(data['atom_types'],At,data['fp']);
        #3.4 获取上三角所有边的真实标签
        i_idx,j_idx=np.triu_indices(data['A0'].shape[0],k=1);
        targets=data['A0_labels'][i_idx,j_idx];
        #3.5 模型预测
        logits=edge_denoiser(torch.tensor(features));
        #3.6 计算损失
        loss=loss_fn(logits,torch.tensor(targets,dtype=torch.long));
        #3.7 反向传播更新参数
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        total_loss+=loss.item();
    avg_loss=total_loss/len(all_data);
    loss_history.append(avg_loss);
    #每10轮打印loss
    if epoch%10==0 or epoch==num_epochs-1:
        print(f"Epoch{epoch:3d},平均loss={avg_loss:.4f}");

####
#4.画loss曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4));
plt.plot(loss_history,label='Training Loss');
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.title('Edge Denoiser Training Loss');
plt.legend();
plt.grid(True);
#用绝对路径避免Windows路径问题
save_dir=os.path.abspath(os.path.join(script_dir,'../images/DAY7'));
os.makedirs(save_dir,exist_ok=True);
plt.savefig(os.path.join(save_dir,'loss_curve.png'),dpi=150,bbox_inches='tight');
print(f"\nLoss曲线已保存到 images/DAY7/loss_curve.png");

#5.保存loss数据后续分析用
np.save(os.path.join(save_dir,'loss_history.npy'),loss_history);
print(f"训练完成，loss数据已保存");