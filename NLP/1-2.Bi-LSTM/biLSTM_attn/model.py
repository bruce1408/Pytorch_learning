import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import const
from torchsummary import summary


class bilstm_attn(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embed_dim, bidirectional, dropout, use_cuda,
                 attention_size, sequence_length):
        super(bilstm_attn, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=const.PAD)  # embed 从1开始，0的话补0
        self.lookup_table.weight.data.uniform_(-1., 1.)  # 归一化到-1 - 1之间

        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, self.layer_size, dropout=self.dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = attention_size
        if self.use_cuda:
            self.w_omega = torch.randn(self.hidden_size * self.layer_size, self.attention_size).cuda()
            self.u_omega = torch.randn(self.attention_size).cuda()
        else:
            self.w_omega = torch.randn(self.hidden_size * self.layer_size, self.attention_size)  # 32 * 16
            self.u_omega = torch.randn(self.attention_size)  # 16

        self.label = nn.Linear(hidden_size * self.layer_size, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size) - > 16 , 16 , 32 * 2

        # -> 16 * 16, 32 * 2
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))  # [16 * 16 , 16]
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        print('the w_omega is: ', self.w_omega)
        return attn_output

    def forward(self, input_sentences, batch_size=None):
        input = self.lookup_table(input_sentences)  # 16 * 16 * 64
        input = input.permute(1, 0, 2)  # [16, 16 ,64] -> seq_len, batch_size, embedding_size

        if self.use_cuda:
            h_0 = torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda()
            c_0 = torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda()
        else:
            h_0 = torch.zeros(self.layer_size, self.batch_size, self.hidden_size)
            c_0 = torch.zeros(self.layer_size, self.batch_size, self.hidden_size)
        # print(h_0.shape, c_0.shape)  # torch.Size([2, 16, 32]) torch.Size([2, 16, 32])

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        # print(lstm_output.shape)  # torch.Size([16, 16, 64])
        # print(final_cell_state.shape)  # torch.Size([2, 16, 32])
        # print(final_hidden_state.shape)  # torch.Size([2, 16, 32])
        attn_output = self.attention_net(lstm_output)
        logits = self.label(attn_output)
        return logits


if __name__ == "__main__":
    batch_size = 16
    seed = 1111
    cuda_able = True
    save = './bilstm_attn_model'
    data = './Dataset/corpus.pt'
    dropout = 0.5
    embed_dim = 64
    hidden_size = 32
    bidirectional = True
    attention_size = 16
    sequence_length = 16
    output_size = 6
    vocab_size = 9277
    # use_cuda = torch.cuda.is_available() and cuda_able
    use_cuda = False

    net = bilstm_attn(batch_size, output_size, hidden_size, vocab_size, embed_dim, bidirectional, dropout, use_cuda,
                      attention_size, sequence_length)

    toydata = torch.randint(0, 9277, (16, 16))
    print(toydata.shape)
    out = net(toydata)
    print(out.shape)
