# DiffMS-Mini 第二周实验总结报告

## 一、实验成果总览

| 天数 | 任务 | 模型 | 数据 | 关键结果 |
|------|------|------|------|---------|
| Day8 | 解码器预训练 | EdgeDenoiser MLP (2073->512->256->5) | MOSES 2000条 | loss 0.44->0.3, validity 92% |
| Day9 | 编码器预训练 | SpecEncoder MLP (1000->512->256->2048) | CANOPUS 1000条 | loss 0.69->0.085, Tanimoto 0.175 |
| Day10/11 | 端到端微调 | 联合模型 | CANOPUS 500条 | loss 1.89->0.384, validity 100% |
| Day12 | 消融实验 | 4组对比 | CANOPUS 200+50条 | 预训练提升validity和tanimoto |
| Day13 | 可视化+分析 | 汇总 | 全部 | 完成3张对比图+分析报告 |

## 二、数据来源一览

| 数据集 | 对应脚本 | 用途 | 状态 |
|--------|---------|------|------|
| MOSES | `00_download_fp2mol_data.sh` | Decoder预训练 | 已下载 |
| HMDB | `00_download_fp2mol_data.sh` | Decoder预训练(备用) | 已下载 |
| COCONUT | `00_download_fp2mol_data.sh` | Decoder预训练(备用) | 已下载 |
| DSSTox | `00_download_fp2mol_data.sh` | Decoder预训练(备用) | 已下载 |
| CANOPUS | `01_download_canopus_data.sh` | Encoder预训练+端到端 | 已下载 |
| MassSpecGym | `02_download_msg_data.sh` | 评测(备用) | 待下载 |

## 三、核心发现

1. **分子式约束极其有效**：
   固定原子列表后Validity达92%（真实指纹条件）和100%（谱编码条件）
2. **MLP解码器是核心瓶颈**：
   边独立预测无法处理同分异构体；升级到Graph Transformer是突破的关键
3. **预训练确实有价值**：
   消融实验证明预训练比从头训练收敛更快、效果更好
4. **多候选采样可提升命中率**：
   Top-10 Tanimoto > Top-1 Tanimoto，验证了"一对多"问题的解决方案

## 四、失败原因分析

| 原因 | 影响 | 解决方案 |
|------|------|---------|
| MLP容量不足 | Tanimoto接近随机水平 | 升级Graph Transformer (DiGress) |
| 数据量小 | 1400倍差距 | 扩大至万级以上 |
| 信息瓶颈 | 谱编码中丢失结构信息 | 接入MIST编码器 |
| 缺少MCES评估 | 无法衡量子结构匹配 | 添加MCES指标 |

## 五、后续工作方向

1. 升级编解码器架构（MLP -> Graph Transformer / MIST）
2. 扩大训练数据规模（k级 -> 万级 -> 百万级）
3. 添加完整评测指标（MCES, Top-k accuracy等）
4. 接入完整MassSpecGym基准测试
5. 尝试公式预测模块（SIRIUS integration）