# DAY8:2026-7-22
08_decoder_pretrain.py
---
[TOC]
# 学习目标
**“实现 decoder 小规模预训练”**
>写一个完整的训练脚本，让解码器学会“从指纹还原分子”


## 数据来源
| 项目 | 内容 |
|------|------|
| 数据集名称 | MOSES 分子结构数据库 |
| 源码仓库脚本 | data_processing/00_download_fp2mol_data.sh |
| 原始数据文件 | data/fp2mol/raw/dataset_v1.csv（81MB, 190万分子） |
| 实验使用 | data/smiles_2k.txt（提取前2000条SMILES） |
# 完成任务
**1.“用 1k～10k 分子训练 fingerprint→graph diffusion decoder”；**
>数据量从之前的小样本（5 个分子）升级到正式规模,输入是指纹（2048位），输出是分子结构（邻接矩阵），训练解码器。

**2.“记录 CE loss、validity”。**
>保存两个指标：损失值 + 生成分子的有效性;
新建一个训练脚本文件,记录训练过程中的 loss 曲线和有效性变化。

## validity
>validity 是指“模型生成的分子中，有多少比例是化学上合法的”。
```
计算流程
1. 从模型中采样 10～20 个分子（从随机图开始，逐步去噪）
2. 对每个生成的邻接矩阵，用 adj_to_mol 转成 RDKit 分子
3. 用 Chem.SanitizeMol() 检查是否合法
4. 统计：合法分子数 / 总采样数 = validity
```

## 具体实现
```
* 第一步：数据加载
把 5 个硬编码分子 → 从 smiles_2k.txt 读 2000 个分子
加 RDKit 过滤，无效的跳过
```
```
* 第二步：数据预处理
对每个 SMILES：build_adj → get_fingerprint → get_atom_one_hot → 存起来
这一步相当于把所有分子提前算好，训练时直接拿来用
```
```
* 第三步：训练循环
沿用 Day 7 的加噪→预测→算loss 流程
但改成每个 epoch 遍历全部 2000 个分子（而不是 5 个）
```
```
* 第四步：加 Validity 评估
每 N 轮（比如每 10 轮），采样一批分子
用 adj_to_mol 检查能否生成合法分子
统计比例 → validity
```
---
**day8_pretrain_log.csv**
记录每轮的 loss 和每 10 轮的 validity
**loss 随 epoch 下降的曲线图**
![](/images/DAY8/loss_curve.png)
