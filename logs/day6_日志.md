# DAY6_Log：2026-7-13
[TOC]
## 学习目标
**"理解图解码器：输入 A_t、X、condition y，预测 A_0"**
>q_sample函数实现了从A0->At，解码器要实现从At->A0
## 完成任务
**"实现简化 Graph Transformer/MPNN edge denoiser，用交叉熵训练边类型"**

### 1.什么是MLP（多层感知机）
MLP就是一个"输入一堆数，通过层层计算，输出另一堆数"的机器。

把MLP想象成一个加工机器：
```
输入：[原子i类型(10个数), 原子j类型(10个数), 噪声边类型(5个数), 指纹(2048个数)]
       ↓
第1层线性层 + ReLU（去除非关键信息）
       ↓
第2层线性层 + ReLU（提炼更高层的模式）
       ↓
第3层线性层（输出5个数，分别代表5种键类型的"得分"）
       ↓
输出：[无键得分, 单键得分, 双键得分, 三键得分, 芳香键得分]
```
一开始这些参数是随机初始化的，所以输出全是乱猜的。

### 2.代码文件：06_graph_decoder.py
#### 2.1 创建MLP模型
```python
edge_denoiser=nn.Sequential(
    nn.Linear(2073,512),
    nn.ReLU(),
    nn.Linear(512,256),
    nn.ReLU(),
    nn.Linear(256,5),
);
```
输入2073维 = 原子i类型(10) + 原子j类型(10) + 噪声边类型(5) + 指纹(2048)

#### 2.2 原子类型映射函数
```python
atom_type_map={'C':0,'N':1,'O':2,'S':3,'P':4,'Cl':5,'Br':6,'F':7,'Si':8};
def get_atom_types(atom_symbols):
    n=len(atom_symbols);
    types=np.zeros((n,10),dtype=np.float32);
    for i,s in enumerate(atom_symbols):
        if s in atom_type_map:
            types[i,atom_type_map[s]]=1.0;
        else:
            types[i,9]=1.0;
    return types;
```

### 3.对解码器的理解
解码器不是一次看整张图，而是每条边独立判断。但因为有fingerprint（全局信息）作为输入，每条判断都能参考整张图。

### 4.遇到的问题
1. PyTorch的模型必须用类或者nn.Sequential定义
2. 一开始不理解MLP，后来用"加工机器"类比才搞明白
