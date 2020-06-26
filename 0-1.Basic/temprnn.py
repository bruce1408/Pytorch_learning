import torch
"""
使用torch.nn.RNN函数来是实现输入hello，输出ohlol
"""
batch_size = 1
input_size = 4
hidden_size = 6
num_layers = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
# 输入hello， 输出的是 ohlol
x_data = [1, 0, 2, 2, 3]  # -> hello
y_data = [3, 1, 2, 3, 2]  # -> ohlol

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, num_layers=1):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


net = Model(input_size, batch_size, hidden_size, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0
    optimizer.zero_grad()  # 清零梯度准备计算
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()  # 反向传播
    optimizer.step()  # 参数更新

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    # print(type(idx))
    result = [idx2char[c] for c in idx.squeeze()]
    print('pred: ', ''.join(result), end='')
    print(', epoch %d loss= %.4f'%(epoch+1, loss.item()))
print('learning finished!')