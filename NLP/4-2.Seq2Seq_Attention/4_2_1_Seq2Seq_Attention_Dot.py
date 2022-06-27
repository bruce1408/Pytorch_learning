# Reference : https://github.com/hunkim/PyTorchZeroToAll/blob/master/14_2_seq2seq_att.py
# https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
dtype = torch.FloatTensor
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch Dataset size is short than time steps
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

# Parameter
n_hidden = 128
word_list = " ".join(sentences).split()
word_list = list(set(word_list))  # 去重
word2index = {w: i for i, w in enumerate(word_list)}  # word2index
index2word = {i: w for i, w in enumerate(word_list)}
n_class = len(word2index)  # vocab list n_class = 11


def make_batch(sentences):
    """
    生成batch—size数据
    :param sentences:
    :return:
    """
    input_batch = [np.eye(n_class)[[word2index[n] for n in sentences[0].split()]]]  # input batch 单词的onehot编码结果
    output_batch = [np.eye(n_class)[[word2index[n] for n in sentences[1].split()]]]
    target_batch = torch.LongTensor([[word2index[n] for n in sentences[2].split()]])
    # make tensor
    return torch.Tensor(input_batch), torch.Tensor(output_batch), target_batch


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # [11, 128]
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # [11, 128]

        # Linear for attention 简单线性连接
        self.attn = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden * 2, n_class)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, enc_inputs, hidden, dec_inputs):
        enc_inputs = enc_inputs.transpose(0, 1)  # enc_inputs: [seq_len, batch_size, n_class]=[5, 1, 11]
        dec_inputs = dec_inputs.transpose(0, 1)  # dec_inputs: [seq_len, batch_size, n_class]=[5, 1, 11]
        """
        encoder 部分没有做任何处理,就是通过一个RNN网络出来.得到context vector.
        """
        # enc_outputs = [5, 1, 128], enc_hidden = [1, 1, 128]
        enc_outputs, enc_hidden = self.enc_cell(enc_inputs, hidden)  # [seq, batch, hidden], [1, batch, hidden]

        trained_attn = []
        hidden = enc_hidden  # [1, 1, 128]
        n_step = len(dec_inputs)  # n_step = 5
        outputs = torch.empty([n_step, 1, n_class])  # 初始化model=[5, 1, 11]

        """
        decoder 部分首先对每个序列长度遍历,初始hidden为encoder_hidden,通过RNN网络得到 decoder_output 和 hidden,
        decoder_output 和 encoder_outputs 进行点乘, 结果进行softmax归一化,得到 attention_weights,这个结果再和encoder_outputs
        进行batch乘法,得到的是context vector, context 和 decoder_output cat合并操作后,通过一个线性层得到当前序列的输出.
        """
        for i in range(n_step):
            # dec_output[1, 1, 128] hidden = [1, 1, 128], hidden = [1, 1, 128],hidden 初始化使用encoder_hidden
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)

            attn_weights = self.get_att_weight(dec_output, enc_outputs)  # attn_weights : [1, 1, n_step]
            # print('atten_weight is: ', attn_weights)

            # [1,1,5] x [1,5,128] = [1,1,128]
            context = attn_weights.bmm(enc_outputs.transpose(0, 1))  # 得到的新context vector

            context = context.squeeze(1)  # [1, num_directions(=1) * n_hidden] = [1, 128]

            dec_output = dec_output.squeeze(0)  # dec_output :[1, 128]
            
            # att 和 decoder_output cat合并操作
            outputs[i] = self.out(torch.cat((dec_output, context), 1))  # context vector 和 decoder_output 生成的输出

            trained_attn.append(attn_weights.squeeze().data.numpy())

        # model = [5, 11], trained_attn(list) = [5, 5]
        return outputs.transpose(0, 1).squeeze(0), trained_attn

    def get_att_weight(self, dec_output, enc_outputs):  # get attention weight one 'dec_output' with 'enc_outputs'
        """
        attention 机制，encode所有时刻的最后一层输出和当前时刻的decode输入进行一个计算
        :param dec_output: 当前时刻的decode输入 [1, 1, 128]
        :param enc_outputs: encode所有时刻的输出 [5, 1, 128]
        :return:
        """
        n_step = len(enc_outputs)
        attn_scores = torch.zeros(n_step)  # attn_scores : [n_step]
        # 对当前时间步的decoder_output 和 enc_outputs 每一个元素分别点乘
        for i in range(n_step):
            attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])  # decoder_output, encoder_outputs 点乘

        # Normalize scores to weights in range 0 to 1 归一化 attn_scores = [5]
        # print('F is: ', F.softmax(attn_scores).view(1, 1, -1))
        # print("S is", self.softmax(attn_scores).view(1, 1, -1))
        # return F.softmax(attn_scores).view(1, 1, -1)
        return self.softmax(attn_scores).view(1, 1, -1)

    def get_att_score(self, dec_output, enc_output):  # enc_outputs [batch_size, num_directions(=1) * n_hidden]
        score = self.attn(enc_output)  # score : [batch_size, n_hidden] = [1, 128]
        return torch.dot(dec_output.view(-1), score.view(-1))  # inner product make scalar value


input_batch, output_batch, target_batch = make_batch(sentences)

hidden = torch.zeros(1, 1, n_hidden)  # [1, 1, 128]

model = Attention()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train, input_batch=[1,5,11], target_batch=[1,5], output_batch=[1,5,11]
for epoch in range(2000):
    optimizer.zero_grad()
    output, _ = model(input_batch, hidden, output_batch)

    loss = criterion(output, target_batch.squeeze(0))
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Test
test_batch = [np.eye(n_class)[[word2index[n] for n in 'SPPPP']]]
test_batch = torch.Tensor(test_batch)

predict, trained_attn = model(input_batch, hidden, test_batch)
predict = predict.data.max(1, keepdim=True)[1]
print(sentences[0], '->', [index2word[n.item()] for n in predict.squeeze()])

# Show Attention
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.matshow(trained_attn, cmap='viridis')
ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
plt.savefig('./img.jpg')
plt.show()