import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLES_DEVICES'] = '0, 2,3'
if torch.cuda.is_available():
    print("multi cuda")
print("torch version: ", torch.__version__)

dtype = torch.FloatTensor
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch Dataset size is short than time steps
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

# Transformer Parameters
# Padding Should be Zero
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
number_dict = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5
tgt_len = 5

d_model = 512  # Embedding Size 词向量的维度
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


def randomSeed(SEED):
    seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


SEED = 1234
randomSeed(SEED)


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])  # [6, 512]
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i 偶数列
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 奇数列
    return torch.FloatTensor(sinusoid_table)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token，因为k的pad没有任何意义，所以这里不计算q为pad的情况
    # [batch_size, 1, len_k] one is masking
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    # 这里维度再进行扩展一下[batch, 1, len_k] -> [batch, len_q, len_k] = [1, 5, 5] 重复复制
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q = [1, 8, 5, 64], attn_mask = [1, 8, 5, 5]
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # Fills elements of self tensor with value where mask is one. 填充mask 是1的地方换成很小的一个数字，这样softmax之后就是0，可以忽略
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # [1, 8, 5, 5] 和 attn_mask 没变
        context = torch.matmul(attn, V)  # [1, 8, 5, 64] 和 Q 没变
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 512, 512
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 512, 512
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 512, 512

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)

        # (B, S, D) -proj-> (B, S, D) -split-> [B, S, H, W] -trans-> [B, H, src_len, W] = [1, 8, 5, 64]
        # q_s:[batch_size, n_heads, len_q, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # k_s:[batch_size, n_heads, len_k, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # v_s:[batch_size, n_heads, len_k, d_v]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attention mask 计算部分 [batch_size, n_heads, len_q, len_k] = [1, 8, 5, 5]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # context:[batch_size, len_q, n_heads, d_v] = [1, 5, 8, 64]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, n_heads * d_v)  # [1, 5, 8*64 = 512]
        output = nn.Linear(n_heads * d_v, d_model)(context)

        # output: [batch_size x len_q x d_model]
        return nn.LayerNorm(d_model)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return nn.LayerNorm(d_model)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs to same Q, K, V = [batch, src_len], enc_self_attn_mask = [batch, q_len, k_len]
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size, len_q, d_model]

        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # 第一部分decoder基于掩码的注意力机制
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        # decoder 和 encoder 级联部分的柱注意力机制
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        # 前向神经网络
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # 词向量
        self.src_emb = nn.Embedding(src_vocab_size, d_model)

        # get_sinusoid_encoding_table 返回的是一个[6, 512]的torch的tensor
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(src_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):  # enc_inputs : [batch_size x source_len]

        # pos是按照索引取值, 可以直接换成enc_inputs, enc_embed = [1, 5, 512]
        enc_embed = self.src_emb(
            enc_inputs) + self.pos_emb(torch.LongTensor([[1, 2, 3, 4, 0]]))

        # enc_self_attn_mask = [batch_size, len_q, len_k]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        enc_self_attns = []
        for layer in self.layers:
            enc_embed, enc_self_attn = layer(enc_embed, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_embed, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(tgt_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    # dec_inputs : [batch_size x target_len]
    def forward(self, enc_inputs, dec_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(
            dec_inputs) + self.pos_emb(torch.LongTensor([[5, 1, 2, 3, 4]]))

        # dec padding mask，如果是有0的位置，说明是有pad，那么对应位置设置为true
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        # [batch_size, len_q, len_k]
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        # 这里是两个mask相加的, 然后结果和0进行比较,如果大于0的话,那么返回1, 否则,返回的是0
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # dec 和 enc 部分的联合mask，encoder级联的部分
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs,
                                                             enc_outputs,
                                                             dec_self_attn_mask,
                                                             dec_enc_attn_mask)

            dec_self_attns.append(dec_self_attn)

            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(enc_inputs, dec_inputs, enc_outputs)

        # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


model = Transformer()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    loss = criterion(outputs, target_batch.contiguous().view(-1))
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()


def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))  # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(),
                       fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()


# Test
predict, _, _, _ = model(enc_inputs, dec_inputs)
predict = predict.data.max(1, keepdim=True)[1]
print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

print('first head of last state enc_self_attns')
showgraph(enc_self_attns)

print('first head of last state dec_self_attns')
showgraph(dec_self_attns)

print('first head of last state dec_enc_attns')
showgraph(dec_enc_attns)
