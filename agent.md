# DiffMS-Mini Agent Log

## 指导原则
导师角色：以导师身份全程指导学生，引导学生思考，解释"为什么"，让学生学到知识、思路和方法，而非直接给出代码。学生有 C/C++/Java 基础，Python 零基础。

## 代码风格规范
1. 分号结尾：每行语句末尾加 ;（保留C/Java习惯）
2. 中文注释：每个代码块解释"做什么"和"为什么"
3. 版块分隔：大版块用 ###，子版块用 ####，编号用中文数字
4. print用中文
5. 注释风格：# 后空一格
6. import单独成块
7. 函数前 #### 一行功能描述
8. 不使用 ===== 作为标题分隔符

## 当前实时进度（更新于 2026-08-09）
当前 DiffMS-Mini 主线已经完成 Day 1-13 的代码、日志、主要图表和阶段性实验结果整理。Day14 还没有正式完成；老师已经布置新的后续任务，明天开始进入“根据已有结果规划下一步实验/改进方案”的阶段。

目前可向老师汇报的核心结论：pipeline 已经跑通，可以从质谱数据经过谱编码器、图解码器生成分子；生成分子的合法率较高，但结构相似度仍然很低，说明当前主要瓶颈不是合法性，而是“谱图条件没有有效约束生成正确结构”。下一步需要围绕条件传递、采样多样性、解码器能力和数据规模做改进。

## 项目完成状态
Day 1-13 已完成；Day14 和老师新任务待继续推进。

### 第一周 (Day 1-7)
- Day 1: 论文理解 ✅
- Day 2: RDKit基础 ✅
- Day 3: Morgan指纹 ✅
- Day 4: 分子图编解码 ✅
- Day 5: 离散扩散 ✅
- Day 6: 图解码器 ✅
- Day 7: 第一周集成验收 ✅

### 第二周 (Day 8-13)
- Day 8: decoder小规模预训练（MOSES 2000条, loss约0.30, validity约92%） ✅
- Day 9: 谱编码器训练（CANOPUS 1000条, 测试集平均 Tanimoto 0.1752） ✅
- Day 10: 端到端连接（CANOPUS 500条, 30轮, validity 100%, Tanimoto 0.0015） ✅
- Day 11: 端到端微调+多候选采样（Top-1/5/10 Tanimoto 均为0, validity 100%） ✅
- Day 12: 消融实验（4组对比, 图表已生成并修复中文乱码） ✅
- Day 13: 结果可视化+失败分析 ✅

### Day 14 / 后续任务（待完成）
- 整理 README、运行命令、实验表格
- 制作 PPT（10~15页）
- 完成复现报告（8~12页）
- 根据老师反馈确定下一步实验计划
- 明天继续推进老师新布置的任务，具体任务内容需要在新对话/新日志中记录清楚

## Day12 最新消融结果
| 组别 | Validity | Tanimoto |
|------|---------:|---------:|
| A-完整预训练 | 100.00% | 0.0015 |
| B-无预训练 | 100.00% | 0.0015 |
| C-仅解码器预训练 | 100.00% | 0.0037 |
| D-仅编码器预训练 | 100.00% | 0.0015 |

Day12 结论：四组实验合法率均为 100%，但 Tanimoto 都接近 0；仅解码器预训练组略高，但仍然很低。说明当前模型能够生成合法分子，但不能生成与目标谱图对应的正确分子。

## 当前已生成、可汇报给老师的图和表
- outputs/ablation_results.csv：Day12 消融实验结果表
- images/DAY12/ablation_comparison.png：Day12 消融实验柱状图，中文乱码已修复
- Day8/Day9/Day10/Day11 的训练 loss、Validity、Tanimoto 结果可作为阶段性汇报材料

## 数据完整度
| 数据集 | 对应原仓库脚本 | 状态 | 当前用途 |
|--------|----------------|------|----------|
| MOSES | 00_download_fp2mol_data.sh | 已下载(81MB) | Day8 图解码器预训练，抽取 smiles_2k.txt |
| HMDB | 00_download_fp2mol_data.sh | 已下载(1.7GB) | fp2mol 原始数据备用 |
| COCONUT | 00_download_fp2mol_data.sh | 已下载(628MB) | fp2mol 原始数据备用 |
| DSSTox | 00_download_fp2mol_data.sh | 已下载(13个xlsx) | fp2mol 原始数据备用 |
| CANOPUS | 01_download_canopus_data.sh | 已下载(10709条) | Day9-Day12 谱图编码、端到端训练、消融实验 |
| MassSpecGym | 02_download_msg_data.sh | 未使用/待下载 | 暂不纳入当前 mini 复现实验 |

## 交付文件清单
scripts/:
  04_graph_utils.py  05_diffusion.py  06_graph_decoder.py  07_train.py
  08_decoder_pretrain.py  09_spectrum_encoder.py  10_diffms_mini.py
  11_finetune.py  11_sample.py  12_ablation.py  13_visualization.py

notebooks/:
  day8_decoder_pretrain.ipynb  day9_spectrum_encoder.ipynb
  day10_end2end.ipynb  day11_finetune_sample.ipynb
  day12_ablation.ipynb  day13_visualization.ipynb

logs/:
  第一周周报.md  第二周周报.md  day2~day13日志.md
  day8~day10训练日志  day10~day13分析报告

checkpoints/:
  spec_encoder.pth  edge_denoiser_2000.pth
  spec_encoder_day11.pth  edge_denoiser_day11.pth

outputs/:
  ablation_results.csv  experiment_summary.md

## 老师新任务记录（2026-08-09）
老师反馈：论文 Table 1 里的对比方法和指标需要自己跑一下；可以先把论文表格里的每一个方法的实验结果跑出来，再根据结果继续规划。老师提到数据集已经有标签、文件结构可以观察，当前任务属于“数据验证/对比实验复现”部分。

### 老师聊天记录中提到的关键要求
- 对照论文 Table 1，关注 DiffMS 与其他 baseline 方法的 Top-1 / Top-10 指标。
- 需要关注的指标包括 Accuracy、MCES、Tanimoto，其中 Accuracy 越高越好，MCES 越低越好，Tanimoto 越高越好。
- 可以先使用家里电脑尝试；如果计算资源不够，可以之后租服务器或使用学校服务器。
- 老师说论文给了多个数据集，约 4G，数据集已经打好标签，可以先观察数据集结构。
- 当前要先把 Table 1 里面每个方法的实验结果跑出来或整理出可复现流程。

### 明天优先任务
1. 先确认论文原仓库里是否提供 Table 1 的 evaluation / baseline 运行脚本。
2. 先不要直接大规模训练，先阅读 README、scripts、config、evaluation 相关文件，确认每个指标怎么计算。
3. 先用已经下载的 CANOPUS / MassSpecGym 或老师给的数据集做小样本验证，跑通 Top-k、Tanimoto、MCES、Accuracy 的计算流程。
4. 整理一个“论文 Table 1 方法-指标-是否能本地跑-需要什么数据/权重”的表。
5. 如果本地跑不动，再准备服务器运行方案，包括环境、数据路径、运行命令和预计耗时。
