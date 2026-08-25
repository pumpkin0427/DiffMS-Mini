# DAY12:2026-7-31
12_ablation.py
---
[TOC]
# 学习目标
**"评测与消融"**
>比较有预训练和无预训练的差异，分析预训练的价值

## 数据来源
| 项目 | 内容 |
|------|------|
| 数据集名称 | CANOPUS 谱-结构配对数据库 |
| 源码仓库脚本 | `data_processing/01_download_canopus_data.sh` |
| 原始数据文件 | `data/canopus/labels.tsv` + `subformulae/subformulae_default/*.json`（10709条谱） |
| 实验使用 | 200条训练 + 50条测试 |

## 四组消融实验设计
| 组别 | 编码器 | 解码器 | 目的 |
|------|-------|--------|------|
| A-完整预训练 | Day9预训练 | Day8预训练 | 基线：最佳性能 |
| B-无预训练 | 随机初始化 | 随机初始化 | 负对照：看预训练的价值 |
| C-仅解码器 | 随机初始化 | Day8预训练 | 看编码器预训练的贡献 |
| D-仅编码器 | Day9预训练 | 随机初始化 | 看解码器预训练的贡献 |

# 完成任务
**"比较无预训练、decoder预训练、encoder预训练的差异"**

## 消融实验流程
```
* 第一步: 200条数据快速训练（20-30轮）
* 第二步: 50条测试集评估validity + Tanimoto
* 第三步: 画出四组柱状图对比
* 第四步: 保存results到 outputs/ablation_results.csv
```

## 预期结论
- 完整预训练组的validity和Tanimoto最高
- 仅解码器预训练组 > 仅编码器预训练组（解码器对分子图生成更重要）
- 无预训练组最低（验证预训练的必要性）

## 交付物
- [x] scripts/12_ablation.py — 消融实验脚本
- [x] outputs/ablation_results.csv — 消融结果表
- [x] images/DAY12/ablation_comparison.png — 消融对比柱状图
