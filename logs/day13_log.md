# DAY13:2026-7-31
13_visualization.py
---
[TOC]
# 学习目标
**"结果可视化与失败分析"**
>汇总Day8-12的所有实验结果，画对比图，分析失败原因

# 完成任务
**"画loss曲线、Top-k best Tanimoto、示例分子图；分析错误原因"**

## 可视化内容
| 图表 | 内容 | 文件名 |
|------|------|--------|
| Loss对比曲线 | Day8/9/10三阶段loss下降对比 | loss_comparison.png |
| 指标对比 | Validity + Tanimoto柱状图 | metrics_comparison.png |
| 模型架构图 | 三阶段模型结构示意图 | architectures.png |
| Top-k柱状图 | Day11多候选采样结果 | topk_tanimoto.png |
| 消融对比 | Day12四组消融柱状图 | ablation_comparison.png |

## 失败分析
```
* 失败原因1: MLP解码器容量不足
  MLP对每条边独立预测，缺少化学键之间的全局依赖信息
  同分异构体有无数合法排列，MLP无法区分目标结构

* 失败原因2: 数据量差距
  论文解码器预训练: 2.8M 指纹-结构对
  本实验: 2000条（差距1400倍）

* 失败原因3: 信息瓶颈
  谱图（1000维bin）→ 编码器 → 2048维条件向量
  压缩过程丢失大量结构信息

* 后续方向:
  (1) 升级解码器为Graph Transformer（DiGress架构）
  (2) 接入MIST编码器
  (3) 扩大数据规模至万级以上
  (4) 引入MCES评估
```

## 交付物
- [x] scripts/13_visualization.py — 可视化 + 分析报告
- [x] images/DAY13/loss_comparison.png
- [x] images/DAY13/metrics_comparison.png
- [x] images/DAY13/architectures.png
- [x] images/DAY13/topk_tanimoto.png + ablation_comparison.png
- [x] logs/day13_summary.md — 实验总结Markdown报告
- [x] logs/day13_analysis.txt — 失败分析报告
