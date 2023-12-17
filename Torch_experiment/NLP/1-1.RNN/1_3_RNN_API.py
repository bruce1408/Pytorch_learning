# import torch
# """
# 使用torch.nn.RNN函数来是实现输入hello，输出ohlol
# """
# batch_size = 1
# input_size = 4
# hidden_size = 4
# num_layers = 1
# seq_len = 5
#
# idx2char = ['e', 'h', 'l', 'o']
# # 输入hello， 输出的是 ohlol
# x_data = [1, 0, 2, 2, 3]  # -> hello
# y_data = [3, 1, 2, 3, 2]  # -> ohlol
#
# one_hot_lookup = [[1, 0, 0, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]]
#
# x_one_hot = [one_hot_lookup[x] for x in x_data]
#
# inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
# labels = torch.LongTensor(y_data).view(-1, 1)
#
#
# class Model(torch.nn.Module):
#     def __init__(self, input_size, batch_size, hidden_size, num_layers=1):
#         super(Model, self).__init__()
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
#
#     def forward(self, input):
#         hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
#         out, _ = self.rnn(input, hidden)
#         print(out.shape)
#         return out.view(-1, self.hidden_size)
#
#
# net = Model(input_size, batch_size, hidden_size, num_layers)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
#
# for epoch in range(15):
#     loss = 0
#     optimizer.zero_grad()  # 清零梯度准备计算
#     outputs = net(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()  # 反向传播
#     optimizer.step()  # 参数更新
#
#     _, idx = outputs.max(dim=1)
#     idx = idx.Dataset.numpy()
#     print('pred: ', "".join(idx2char[idx.item()]), end='')
#     print(' ,epoch %d loss= %.4f'%(epoch+1, loss.item()))




# Lab 12 RNN
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)  # reproducibility


num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
num_layers = 1  # one-layer rnn
idx2char = ['h', 'i', 'e', 'l', 'o']  # - > 0, 1, 2, 3, 4

# Teach hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]   # input: hihell

one_hot_lookup = [[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]]

x_one_hot_1 = [one_hot_lookup[x] for x in x_data[0]]

# x_one_hot_1 = [[[1, 0, 0, 0, 0],   # h 0
#               [0, 1, 0, 0, 0],   # i 1
#               [1, 0, 0, 0, 0],   # h 0
#               [0, 0, 1, 0, 0],   # e 2
#               [0, 0, 0, 1, 0],   # l 3
#               [0, 0, 0, 1, 0]]]  # l 3

y_data = [1, 0, 2, 3, 3, 4]    # ihello

# As we have one batch of samples, we will change them to variables only once
inputs = torch.Tensor(x_one_hot_1).view(-1, sequence_length, input_size)

# inputs = Variable(torch.Tensor(x_one_hot))
labels = torch.LongTensor(y_data)


class RNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Reshape input shape=[batch_size, seq_len, input_size]
        x.view(x.size(0), self.sequence_length, self.input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)

        out, _ = self.rnn(x, h_0)
        return out.view(-1, num_classes)


# Instantiate RNN model
rnn = RNN(num_classes, input_size, hidden_size, num_layers)
print(rnn)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    outputs = rnn(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data), end='')
    print(" Predicted string: ", ''.join(result_str))

print("Learning finished!")
