"""
    Reference:
    https://github.com/jadore801120/attention-is-all-you-need-pytorch
    https://github.com/JayParks/transformer, https://github.com/dhlee347/pytorchic-bert
    using bert to predict next sentence
"""
import os
import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# BERT Parameters
maxlen = 30     # 同一个batch里面所有的样本长度都必须是相同的
batch_size = 6  # 最多预测的mask个单词，因为15%的话，如果长度是100个词，可能会有很多mask，15个，这里设定max_pred之后会少很多
max_pred = 5    # max tokens of prediction
n_layers = 6    # encoder layer 层数
n_heads = 12    # 多少个head
d_model = 768   # 三个维度，word， segment，postion embedding维度
d_ff = 768 * 4  # 4*d_model, FeedForward dimension 全连接神经网络维度
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2  # 一个样本里面是多少句话，论文中是两句话

text = (
    'Hello, how are you? I am Romeo.\n'  # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'  # J
    'Nice meet you too. How are you today?\n'  # R
    'Great. My baseball team won the competition.\n'  # J
    'Oh Congratulations, Juliet\n'  # R
    'Thanks you Romeo'  # J
)


def randomSeed(SEED):

    seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


SEED = 1234
randomSeed(SEED)

# 所有的标点符号全部替换成空格,且大写变小写
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))  # word vocabulary 单词的词典

# 单词对应的index编号，Mask 表示替换的单词
word2num = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

for i, w in enumerate(word_list):
    word2num[w] = i + 4
# 数字对应的单词字典,word2num
number2dict = {i: w for i, w in enumerate(word2num)}
vocab_size = len(word2num)  # 29个词
token_list = list()  # 每个sentence的index的list

for sentence in sentences:
    arr = [word2num[s] for s in sentence.split()]
    token_list.append(arr)


# sample IsNext and NotNext to be same in small batch size


def make_batch():
    batch = []

    # positive表示一个样本的两句话是不是相邻的，否则用negative表示
    positive = negative = 0

    # sample random index in sentences
    while positive != batch_size / 2 or negative != batch_size / 2:

        # 随机选择数据中的两个句子的索引, a 和 b是否相邻可以a+1 == b？
        tokens_a_index, tokens_b_index = randrange(
            len(sentences)), randrange(len(sentences))

        # 得到两个句子a 和 b
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]

        # 在预测下一个句子的任务中, CLS符号将对应的文本语义表示,对两句话用一个SEP符号分割,并分别对两句话附加两个不同的文本向量区分
        input_ids = [word2num['[CLS]']] + tokens_a + \
            [word2num['[SEP]']] + tokens_b + [word2num['[SEP]']]

        # segmeng设置，第一句话全是0，第二句话全是1，[0] * (cls + len_seq+ 1)
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # 开始使用MASK LM, 替换，mask_pred = 5，随机把一句话中15%的token进行替换或者是mask操作, 如果这句话单词个数很短，那么就找到最长的，然后再找到最短的
        # 15 % of tokens in one sentence
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))

        # 特殊字符cls和sep作为mask没有任何意义，所以排除这些特殊的，找到候选的mask的位置，排除了cls和sep之后的input_ids的下标索引位置
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2num['[CLS]'] and token != word2num['[SEP]']]

        # cand_mask_pos = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15] # 不包含CLS 和 SEP 句子的input_ids下标index
        shuffle(cand_maked_pos)  # 随机做mask，所以打乱mask候选位置
        masked_tokens, masked_pos = [], []

        # 只要取前n_pred个单词
        for pos in cand_maked_pos[:n_pred]:

            # 存放下标
            masked_pos.append(pos)

            # 存放索引
            masked_tokens.append(input_ids[pos])

            # 如果小于0.8那么就要替换mask
            if random() < 0.8: 
                input_ids[pos] = word2num['[MASK]']  # make mask
            elif random() > 0.9: # 10% 的概率是替换其他单词这里注意不能替换没有意义的四个单词，必须是有意义的才对。
                index = randint(0, vocab_size - 1) 
                # 如果随机替换的单词是前面4个没有意义的单词，那么重新选择，知道找到有意义的单词为止
                while index < 4:
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index

        # Zero Paddings, max_len = 30，最长的单词个数是30，如果不够30，那么就要补mask
        n_pad = maxlen - len(input_ids)

        # input 和 segment都要同时补0
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens，这里mask也要数目一致，如果不足max_mask, 要补足最大mask的个数
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # 判断这两句话是不是相邻的，positive和negative数目1：1，所以不能超过1半
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens,
                         masked_pos, True])  # IsNext
            positive += 1
        # 同理negative也是一样的
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens,
                         masked_pos, False])  # NotNext
            negative += 1
    return batch


# Proprecessing Finished

def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_k的数据是否等于0,如果等于0的话就返回true,然后生成的是一个atten的mask
    :param seq_q:
    :param seq_k:
    :return:
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # [batch_size, 1, len_k(=len_q)], one is masking
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # [batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def gelu(x):
    """
    Implementation of the gelu activation function by Hugging Face
    https://zhuanlan.zhihu.com/p/302394523
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        # segment(token type) embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        # (seq_len,) -> (batch_size, seq_len)
        pos = pos.unsqueeze(0).expand_as(x)
        input_emb = self.tok_embed(x)  # [6, 30, 768]
        pos_emb = self.pos_embed(pos)  # [6, 30, 768]
        seg_emb = self.seg_embed(seg)
        embedding = input_emb + pos_emb + seg_emb
        return self.norm(embedding)


# 缩放点积函数部分
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):

        # scores : [batch_size, n_heads, len_q(=len_k), len_k(=len_q)] = [6, 12, 30, 30]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # Fills elements of self tensor with value where mask is one，这里是1就忽略计算，用小的数填充即可
        scores.masked_fill_(attn_mask, -1e9)

        # 进行一个归一化操作之后
        attn = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, n_heads, len_q, d_k] = [6, 12, 30, 64]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # k_s: [batch_size x n_heads x len_k x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # v_s: [batch_size x n_heads x len_k x d_v]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attn_mask : [batch_size, n_heads, len_q, len_k] , 把他 repeat n_heads 份
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, n_heads * d_v)

        output = nn.Linear(n_heads * d_v, d_model)(context)

        # output: [batch_size x len_q x d_model]
        return nn.LayerNorm(d_model)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):

        # 第一次的enc_inputs 是 pos + seg + word embedding 三者相加得到的,维度是[6, 30, 768]
        # enc_inputs to same Q, K, V
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)

        # decoder is shared with embedding layer, 输入embed的权重大小
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):

        output = self.embedding(input_ids, segment_ids)

        # [6, 30, 30] 用于计算掩码注意力机制的padding mask问题
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)

        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]

        # it will be decided by first token(CLS) [batch_size, d_model], 只是对第一个token cls进行处理
        h_pooled = self.activ1(self.fc(output[:, 0]))

        # [batch_size, 2]
        logits_clsf = self.classifier(h_pooled)

        # [batch_size, max_pred, d_model] = [6, 5, 768]
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))

        # get masked position from final output of transformer. masking position [batch_size, max_pred, d_model]
        # 从masked_pos中选取对应的output, 因为这里要求姐mask的损失，输出维度和output一样
        h_masked = torch.gather(output, 1, masked_pos)

        # 前向网络部分
        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        # [batch_size, max_pred, n_vocab]
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return logits_lm, logits_clsf


model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# batch中的每一个样本大小是[list_len = 30, 30, 5, 5, T/F] 格式
batch = make_batch()

# inputs_ids seg_ids = [6, 30], masked_pos = [6, 5], isNext = [6,]
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext = torch.LongTensor(input_ids),\
    torch.LongTensor(segment_ids),\
    torch.LongTensor(masked_tokens), \
    torch.LongTensor(masked_pos),\
    torch.LongTensor(isNext)

for epoch in range(1000):
    optimizer.zero_grad()
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)

    # for masked LM
    loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
    loss_lm = (loss_lm.float()).mean()

    # for sentence classification
    loss_clsf = criterion(logits_clsf, isNext)
    loss = loss_lm + loss_clsf
    if (epoch + 1) % 10 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# Predict mask tokens ans isNext
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]
print(text)
print([number2dict[w] for w in input_ids if number2dict[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]),
                               torch.LongTensor([segment_ids]),
                               torch.LongTensor([masked_pos]))

logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ', [pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ', True if logits_clsf else False)
