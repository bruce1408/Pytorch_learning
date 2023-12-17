# https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
"""
主要介绍 pack_padded_sequence 函数 和 pad_packed_sequence 函数的用法
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
import itertools


def flatten(l):
    # 每一个单词分成字母, 并且排序
    return list(itertools.chain.from_iterable(l))


# length= [10, 6, 11]
seqs = ['ghatmasala', 'nicela', 'chutpakodas']

# make <pad> idx 0
vocab = ['<pad>'] + sorted(list(set(flatten(seqs))))  # 词典列表
embedding_size = 3
embed = nn.Embedding(len(vocab), embedding_size)
lstm = nn.LSTM(embedding_size, 5)
vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]  # 字符串转化为index序列
# vectorized_seqs [[5, 6, 1, 15, 10, 1, 14, 1, 9, 1], [11, 7, 2, 4, 9, 1], [2, 6, 16, 15, 13, 1, 8, 12, 3, 1, 14]]

# get the length of each seq in your batch, 获得每个batch的长度
seq_lengths = torch.LongTensor([x for x in map(len, vectorized_seqs)])  # 得到的是[10, 6, 11]的list然后转tensor


# 对每个batch按照最长的lengths，然后进行对齐
seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max()), dtype=torch.long)  # shape是[3 x 11]的零向量
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
    seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

# 按照batch中的长度，从长到短进行排序
seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)  # [11, 10, 6]

# 按照降序进行排列之后的长度值 seq_tensor = [batch, seq_len]
seq_tensor = seq_tensor[perm_idx]

# utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
# Otherwise, give (L,B,D) tensors
seq_tensor = seq_tensor.transpose(0, 1)  # (Batch_size, seq_Len, D)=[3 x 11] -> (seq_Len, Batch_size, D)=[11 x 3]
# 等同于 seq_tensor.permute(1, 0)

# 11 x 3 x 3, 从这里开始进入pack_padded_sequence，输入的是按照长度从长到短的序列和长度列表
embeded_seq_tensor = embed(seq_tensor)

# seq_lengths需要从大到小排列才可以
packed_input = pack_padded_sequence(embeded_seq_tensor, seq_lengths.cpu().numpy())

# throw them through your LSTM (remember to give batch_first=True here if you packed with it)
packed_output, (ht, ct) = lstm(packed_input)  # packed_input 输入是 [27 x 3] 总共是27的有效长度, 输出是27x5

# unpack your output if required  解压缩即可,注意，这里实际上是对output进行pad，但是h或者是c的话，pad是会报错的；
output, _ = pad_packed_sequence(packed_output)

# [seq_len, batch, hidden_dim] = [11 x 3 x 5]
print("Lstm output\n", output.size())
print("Last output \n", ht[-1].size())
print("last hidden Dataset is \n", ht[-1].data)

# 不需要进行 pack_padded , 不需要压缩直接进行lstm单元计算
unpackout, (h, c) = lstm(embeded_seq_tensor)

