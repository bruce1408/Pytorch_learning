import torch
import numpy as np
"""
单个rnncell，其实就是一个rnn单元；
使用torch.nn.RNNCell实现输入hello, 输出ohlol功能
"""
num_epoch = 3
batch_size = 1
input_size = 4
hidden_size = 4

idx2char = ['e', 'h', 'l', 'o']
# 输入hello， 输出的是 ohlol
x_data = [1, 0, 2, 2, 3]  # -> hello
y_data = [3, 1, 2, 3, 2]  # -> ohlol

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

# shape = 5 * 4
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)


class Model(torch.nn.Module):
    def __init__(self, input_size, batch_size, hidden_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = Model(input_size, batch_size, hidden_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(num_epoch):
    loss = 0
    optimizer.zero_grad()  # 清零梯度准备计算
    hidden = net.init_hidden()  # 初始化h_0
    print('train start:, ', end='')
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()  # 反向传播
    optimizer.step()  # 参数更新
    print(' ,epoch %d loss= %.4f'%(epoch+1, loss.item()))
