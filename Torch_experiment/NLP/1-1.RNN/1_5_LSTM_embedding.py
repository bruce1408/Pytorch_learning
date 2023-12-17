import torch
import torch.nn as nn

torch.manual_seed(0)
# parameters
num_layers = 2
hidden_size = 5
batch_size = 1
seq_len = 6
input_size = 5
num_class = 5
embedding_size = 10

idx2char = ['h', 'i', 'e', 'l', 'o']  # - > 0, 1, 2, 3, 4
x_data = [[0, 1, 0, 2, 3, 3]]  # hihell [input sequence]
y_data = [1, 0, 2, 3, 3, 4]  # ihello [output sequence]
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, num_class, batch_size):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)  # 5 * 10
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.num_class = num_class
        self.batch_size = batch_size
        self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        # self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, input):
        emb = self.embedding(input)
        # print(emb.shape)
        emb = emb.view(-1, batch_size, embedding_size)
        h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        output, _, = self.LSTM(emb, (h_0, c_0))
        # self.fc(output.view(-1, hidden_size))
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

