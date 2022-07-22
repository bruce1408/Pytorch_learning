import torch
import torch.nn as nn
from torch.functional import F
# from torchsummary import summary

# 这里的attention是lstm输出output和查询向量q进行的计算，这里的查询向量其实就是隐状态hn


class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.n_hidden = n_hidden
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)  # embedding
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        hidden = final_state.view(-1, self.n_hidden * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] =
        # [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        input = self.embedding(X)  # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]

        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        hidden_state = torch.zeros(1 * 2, len(X), self.n_hidden)

        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1 * 2, len(X), self.n_hidden)

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))

        # output : [batch_size, len_seq, n_hidden]
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)

        # model : [batch_size, num_classes], attention : [batch_size, n_step]
        return self.out(attn_output), attention


if __name__ == "__main__":

    # embedding这里要用int类型，否则报错
    inputs = torch.randint(100, (5, 10))
    print(inputs.type)
    net = BiLSTM_Attention(100, 10, 6, 2)
    print(net)
    outputs = net(inputs)
    print(outputs[0].shape)


