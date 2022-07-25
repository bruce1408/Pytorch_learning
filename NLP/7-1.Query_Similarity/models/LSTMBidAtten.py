import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMAttn(nn.Module):

    def __init__(self, embed_size, label_size, dropout=0.3):
        super(LSTMAttn, self).__init__()
        self.embed = nn.Embedding(embed_size, 100)
        # 双向lstm
        self.lstm = nn.LSTM(100, 100, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(100 * 2, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, label_size)
        self.dropout = nn.Dropout(dropout)
        # self.Sigmoid = nn.Sigmoid() # method1
        self.relu = nn.ReLU()
        self.name = "LSTMBidAtten"

    def attention_net(self, lstm_output, final_state):
        # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        hidden = final_state.view(-1, 200, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] =
        # [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, a, b, lengths_a, lengths_b):
        a = self.embed(a)
        b = self.embed(b)

        a = pack_padded_sequence(a, lengths_a, batch_first=True, enforce_sorted=False)
        b = pack_padded_sequence(b, lengths_b, batch_first=True, enforce_sorted=False)

        hidden_first, (hn1, _) = self.lstm(a)
        hidden_second, (hn2, _) = self.lstm(b)

        hidden_first, _ = pad_packed_sequence(hidden_first, batch_first=True)
        hidden_second, _ = pad_packed_sequence(hidden_second, batch_first=True)

        hidden_first, _ = self.attention_net(hidden_first, hn1)
        hidden_second, _ = self.attention_net(hidden_second, hn2)
        # print(hidden_first.shape)
        outputs_first = self.dropout(self.fc1(hidden_first))
        outputs_second = self.dropout(self.fc1(hidden_second))

        outputs_first = self.fc2(outputs_first)
        outputs_second = self.fc2(outputs_second)

        outputs_first = self.dropout(self.fc3(outputs_first))
        outputs_second = self.dropout(self.fc3(outputs_second))

        outputs = outputs_first + outputs_second
        return outputs


if __name__ == "__main__":
    input1 = torch.randint(0, 5, (3, 5))
    input2 = torch.randint(0, 5, (3, 5))

    input1_len = torch.tensor([2, 1, 3])
    input2_len = torch.tensor([3, 4, 2])

    model = LSTMAttn(5, 3)
    output = model(input1, input2, input1_len, input2_len)
    print(output.shape, output)