"""
# Teach hihell -> ihello

"""
import torch
import torch.nn as nn
torch.manual_seed(0)

num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
num_layers = 1  # one-layer rnn
idx2char = ['h', 'i', 'e', 'l', 'o']  # - > 0, 1, 2, 3, 4

x_data = [[0, 1, 0, 2, 3, 3]]   # input: hihell
one_hot_lookup = [[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data[0]]
y_data = [1, 0, 2, 3, 3, 4]    # ihello

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input):
        h_0 = torch.randn(num_layers, batch_size, self.hidden_size)
        output, h_n = self.gru(input, h_0)
        output = output.view(-1, hidden_size)
        return output


gru = GRU(input_size, hidden_size, num_layers, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gru.parameters(), lr=0.1)

for epoch in range(100):
    outputs = gru(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    resultstr = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data), end='')
    print(" Predicted string: ", ''.join(resultstr))

print("Learning finished!")

