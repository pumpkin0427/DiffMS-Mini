import numpy as np

def get_noise_schedule(T=500):#论文里面默认五百步来打乱
    betas=np.linspace(1e-4,0.02,T);
    #for i in range(betas):
    # alphas.append(1.0-i);
    alphas=1.0-betas;#np可以直接对数组做运算
    alpha_bars=np.cumprod(alphas);
    return alpha_bars;

###
def q_sample(A0,t,alpha_bars):
    A0 = A0.copy();#AI改进建议:防止A0被污染
    alpha=alpha_bars[t];
    #计算骰子的数量
    n=A0.shape[0];
    all=n*(n-1)//2;
    #列表为1的就是被选中要改动的边
    mask=np.random.binomial(1,1-alpha,size=all);
    i_idx,j_idx=np.triu_indices(n, k=1);
    #找到mask中被选中的边的索引值
    modify_idx=np.where(mask==1)[0];
    #把该索引值映射为矩阵中的位置坐标
    modify_i = i_idx[modify_idx];
    modify_j = j_idx[modify_idx];
    #第二个随机过程：随机选一个键进行更改替换
    random_types = np.random.randint(0, 5, size=len(modify_idx))
    # 把选中的位置全部清成 0（无键）
    A0[modify_i, modify_j, :] = 0
    A0[modify_j, modify_i, :] = 0
    # 在清空的位置上填入新的随机键类型
    A0[modify_i, modify_j, random_types] = 1
    A0[modify_j, modify_i, random_types] = 1
    #确保对角线无键状态
    np.fill_diagonal(A0[:, :, 0], 1)
    return A0;

###
def sample_timesteps(batch_size, T):
    #随机选取一个0-499的数进行加噪
    return np.random.randint(0, T, size=batch_size)