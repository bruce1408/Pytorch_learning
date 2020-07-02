import torch
import torch.nn as nn

torch.manual_seed(0)
# parameters
num_layers = 1
hidden_size = 5
batch_size = 1
seq_len = 6
input_size = 5
num_class = 5

idx2char = ['h', 'i', 'e', 'l', 'o']  # - > 0, 1, 2, 3, 4
x_data = [[0, 1, 0, 2, 3, 3]]  # hihell input sequence
y_data = [1, 0, 2, 3, 3, 4]  # ihello output sequence
one_hot_lookup = [[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]]

x_one_hot = [one_hot_lookup[x] for x in x_data[0]]
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data)


class LSTM(nn.Module):
    """
    双向LSTM不一样的就是输出的维度，是 direction * hidden_size, 这里的direction=2
    """
    def __init__(self, input_size, hidden_size, num_layers, seq_len, num_class, batch_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.num_class = num_class
        self.batch_size = batch_size
        # 使用双向LSTM
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * 2, self.num_class)

    def forward(self, input):

        h_0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size)
        output, _, = self.LSTM(input, (h_0, c_0))
        # self.fc(output.view(-1, hidden_size))
        output = output.view(-1, self.hidden_size * 2)
        output = self.linear(output)
        return output.view(-1, self.num_class)


lstm = LSTM(input_size, hidden_size, num_layers, seq_len, num_class, batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

for epoch in range(1000):
    output = lstm(inputs)
    optimizer.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    _, idx = output.max(1)
    idx = idx.data.numpy()

    resultstr = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data), end='')
    print(" Predicted string: ", ''.join(resultstr))
