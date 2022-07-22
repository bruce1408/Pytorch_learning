import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):

    def __init__(self, embed_size, label_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(embed_size, 100)
        self.lstm = nn.LSTM(100, 32, batch_first=True)
        self.output = nn.Linear(32, label_size)
        self.dropout = nn.Dropout(dropout)
        # self.Sigmoid = nn.Sigmoid() # method1
        self.relu = nn.ReLU()

    def forward(self, a, b, lengths_a, lengths_b):
        a = self.embed(a)
        b = self.embed(b)

        a = pack_padded_sequence(a, lengths_a, batch_first=True, enforce_sorted=False)
        b = pack_padded_sequence(b, lengths_b, batch_first=True, enforce_sorted=False)

        hidden_first, (_, _) = self.lstm(a)
        hidden_second, (hn_second, cn_second) = self.lstm(b)

        hidden_first, _ = pad_packed_sequence(hidden_first, batch_first=True)
        hidden_second, _ = pad_packed_sequence(hidden_second, batch_first=True)

        outputs_first = self.output(hidden_first)
        outputs_second = self.output(hidden_second)
        outputs = outputs_first.sum(1) + outputs_second.sum(1)
        return outputs


if __name__ == "__main__":
    input1 = torch.randint(0, 5, (2, 5))
    input2 = torch.randint(0, 5, (2, 5))

    input1_len = torch.tensor([2, 1])
    input2_len = torch.tensor([3, 4])

    model = LSTMModel(5, 3)
    output = model(input1, input2, input1_len, input2_len)
    print(output.shape, output)