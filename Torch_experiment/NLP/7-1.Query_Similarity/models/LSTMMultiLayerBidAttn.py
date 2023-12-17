import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Net(nn.Module):

    def __init__(self, embed_size, label_size, dropout=0.3):
        super(Net, self).__init__()
        self.embed = nn.Embedding(embed_size, 100)
        # 多层双向 attention-lstm
        self.lstm = nn.LSTM(100, 100, num_layers=3, batch_first=True, bidirectional=True)

        # bidirection = 2 * numlayers = 3
        self.fc1 = nn.Linear(100 * 2 * 3, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, label_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.name = "LSTMMultiLayerBidAttn"

    def attention_net(self, lstm_output, final_state):
        # lstm_output.shape = [batch, seqlen, bid*hidden], final_state.shape=[6, 2, 100]
        # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        # print("final_state", final_state.shape)
        hidden = final_state.view(-1, 200, 3)
        # print("lstm: ", lstm_output.shape)
        # print('hidden: ', hidden.shape)
        # print('outshape:', torch.bmm(lstm_output, hidden))
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        # print("atte", attn_weights.shape)
        soft_attn_weights = F.softmax(attn_weights, 1)
        # print('soft: ', soft_attn_weights.shape)

        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] =
        # [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).view(lstm_output.shape[0], -1, 1).squeeze(2)
        # print("context: ", context.shape)
        return context, soft_attn_weights.data.cpu().numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, a, b, lengths_a, lengths_b):
        a = self.embed(a)
        b = self.embed(b)
        # print("embedding: ", a.shape)

        a = pack_padded_sequence(a, lengths_a, batch_first=True, enforce_sorted=False)
        b = pack_padded_sequence(b, lengths_b, batch_first=True, enforce_sorted=False)

        # hidden_first=[3, hidden*bid], hn1=shape [bid * layers, bid=2, hidden]
        hidden_first, (hn1, _) = self.lstm(a)
        hidden_second, (hn2, _) = self.lstm(b)
        # print("hn1 :", hn1.shape)
        hidden_first, _ = pad_packed_sequence(hidden_first, batch_first=True)
        hidden_second, _ = pad_packed_sequence(hidden_second, batch_first=True)

        # print('hiddden-first', hidden_first[0].shape)
        hidden_first, _ = self.attention_net(hidden_first, hn1)
        # print(hidden_first.shape)
        hidden_second, _ = self.attention_net(hidden_second, hn2)

        outputs_first = self.dropout(self.fc1(hidden_first))
        outputs_second = self.dropout(self.fc1(hidden_second))

        outputs_first = self.fc2(outputs_first)
        outputs_second = self.fc2(outputs_second)

        outputs_first = self.dropout(self.fc3(outputs_first))
        outputs_second = self.dropout(self.fc3(outputs_second))

        outputs = outputs_first + outputs_second
        return outputs


if __name__ == "__main__":
    input1 = torch.randint(0, 5, (4, 5))
    input2 = torch.randint(0, 5, (4, 5))

    input1_len = torch.tensor([2, 3, 1, 4])
    input2_len = torch.tensor([3, 4, 2, 2])

    model = Net(5, 3)
    output = model(input1, input2, input1_len, input2_len)
    print(output.shape, output)