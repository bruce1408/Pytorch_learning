import torch
from torch import nn

# ################ RNN 单元 #################
"""
参考 https://blog.csdn.net/SHU15121856/article/details/104387209
使用RNN单元，
"""
# 例子 1
rnn = nn.RNN(100, 10, 1)
print(rnn._parameters.keys())
print(rnn.weight_ih_l0.shape)  # torch.Size([10, 100])
print(rnn.weight_hh_l0.shape)  # torch.Size([10, 10])
print(rnn.bias_ih_l0.shape)  # torch.Size([10])
print(rnn.bias_hh_l0.shape)  # torch.Size([10])

# 例子 2
# 表示feature_len=100, hidden_size=20, 层数=1
rnn = nn.RNN(100, 20, 1, batch_first=True)
# 输入3个样本序列(batch=3), 序列长为10(seq_len=10), 每个特征100维度(feature_len=100)
x = torch.randn(3, 10, 100)
# 传入RNN处理, 另外传入h_0, shape是<层数, batch, hidden_len=20>
out, h = rnn(x, torch.zeros(1, 3, 20))
"""
输出是每一个时刻在空间上最后一层的输出[seq, batch, hidden_size]
ht是最后一个时刻上所有层的记忆单元 [num_layers, batch, hidden_size]
"""
# 输出返回的out和最终的隐藏记忆单元的shape
print('out ', out.shape)  # torch.Size([10, 3, 20])
print('h ', h.shape)  # torch.Size([1, 3, 20])

# ################ RNNCell 一层 #################
# 例子 1
# 表示feature_len=100, hidden_len=20
cell = nn.RNNCell(100, 20)
# 某一时刻的输入, 共3个样本序列(batch=3), 每个特征100维度(feature_len=100)
x = torch.randn(3, 100)
# 所有时刻的输入, 一共有10个时刻, 即seq_len=10
xs = [torch.randn(3, 100) for i in range(10)]
# 初始化隐藏记忆单元, batch=3, hidden_len=20
h = torch.zeros(3, 20)
# 对每个时刻的输入, 传入这个nn.RNNCell计算单元, 还要传入上一时h, 以进行前向计算
for xt in xs:
    h = cell(xt, h)
# 查看一下最终输出的h, 其shape还是<batch, hidden_len>
print('h ', h.shape)  # torch.Size([3, 20])

# 多层RNN Cell 单元
# 例子 2
# 第0层和第1层的计算单元
cell_l0 = nn.RNNCell(100, 30)  # feature_len=100, hidden_len_l0=30
cell_l1 = nn.RNNCell(30, 20)  # hidden_len_l0=30, hidden_len_l1=20

# 第0层和第1层使用的隐藏记忆单元(图中黄色和绿色)
h_l0 = torch.zeros(3, 30)  # batch=3, hidden_len_l0=30
h_l1 = torch.zeros(3, 20)  # batch=3, hidden_len_l1=20

# 原始输入, batch=3, feature_len=100
xs = [torch.randn(3, 100) for i in range(4)]  # seq_len=4, 即共4个时刻

for xt in xs:
    h_l0 = cell_l0(xt, h_l0)
    h_l1 = cell_l1(h_l0, h_l1)

# 图中最右侧两个输出
print(h_l0.shape)  # torch.Size([3, 30])
print(h_l1.shape)  # torch.Size([3, 20])
