import torch
import torch.nn as nn
import torch.nn.functional as F

# """
# 参考资料
# https://zhuanlan.zhihu.com/p/93061413
# """
# # parameters
# vocab_size = 100
# embedding_dim = 20
# n_hidden = 30
# num_classes = 5
#
#
# class BiLSTM_Attention(nn.Module):
#     def __init__(self):
#         super(BiLSTM_Attention, self).__init__()
#
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
#         self.out = nn.Linear(n_hidden * 2, num_classes)
#
#     # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
#     def attention_net(self, lstm_output, final_state):
#         #  hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
#         hidden = final_state.view(-1, n_hidden * 2, 1)
#         attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
#         soft_attn_weights = F.softmax(attn_weights, 1)
#         # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] =
#         # [batch_size, n_hidden * num_directions(=2), 1]
#         context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
#         return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]
#
#     def forward(self, X):
#         input = self.embedding(X)  # input : [batch_size, len_seq, embedding_dim]
#         input = input.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]
#
#         hidden_state = torch.zeros(1 * 2, len(X),
#                                    n_hidden)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
#         cell_state = torch.zeros(1 * 2, len(X), n_hidden)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
#
#         # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
#         output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
#         output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, n_hidden]
#         attn_output, attention = self.attention_net(output, final_hidden_state)
#         return self.out(attn_output), attention  # model : [batch_size, num_classes], attention : [batch_size, n_step]
#
#
# if __name__ == "__main__":
#     inputs = torch.randint(0, 10, (3, 20))
#     net = BiLSTM_Attention()
#     outputs = net(inputs)
#     print(outputs[0])


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class AttnClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def set_embedding(self, vectors):
        self.embedding.weight.data.copy_(vectors)

    def forward(self, inputs, lengths):
        batch_size = inputs.size(1)
        # (L, B)
        embedded = self.embedding(inputs)
        # (L, B, E)
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        out, hidden = self.lstm(packed_emb)
        out = nn.utils.rnn.pad_packed_sequence(out)[0]
        out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        # (L, B, H)
        embedding, attn_weights = self.attention(out.transpose(0, 1))
        # (B, HOP, H)
        outputs = self.fc(embedding.view(batch_size, -1))
        # (B, 1)
        return outputs, attn_weights
