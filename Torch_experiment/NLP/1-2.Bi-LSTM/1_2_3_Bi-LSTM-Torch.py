import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若能使用cuda，则使用cuda
print(device)
dtype = torch.FloatTensor

sentence = (
    'Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation et dolore')


word2idx = {w: i for i, w in enumerate(list(set(sentence.split())))}
idx2word = {i: w for i, w in enumerate(list(set(sentence.split())))}
n_class = len(word2idx)
max_len = len(sentence.split())
n_hidden = 5


def make_batch(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word2idx[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))
        input_batch.append(np.eye(n_class)[input])

    target = torch.LongTensor([word2idx[word] for word in words[:-1]])

    return Variable(torch.Tensor(input_batch)), target
    # return input_batch, target


class BiLSTM(nn.Module):
    """
    h_n 和 c_n表示保存的是每层最后一个time step的状态
    output表示的是最后一层的每个时间步的输出

    """
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Parameter(torch.randn([n_hidden * 2, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, X):
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        hidden_state = torch.zeros(1 * 2, len(X), n_hidden)
        # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = torch.zeros(1 * 2, len(X), n_hidden)

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, n_class]
        return model


input_batch, target_batch = make_batch(sentence)

model = BiLSTM()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10000):
    optimizer.zero_grad()
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    model = model.to(device)
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

predict = model(input_batch).data.max(1, keepdim=True)[1]
print(sentence)
print([idx2word[n.item()] for n in predict.squeeze()])
