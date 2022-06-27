# Defined in Section 5.2.3.3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import sys
sys.path.append("../")
from tools.utils import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from tools.utils import load_reuters, save_pretrained, get_loader, init_weights


# 基于负采样的skipGram模型
class SGNSDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2, n_negatives=5, ns_dist=None):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence)-1):

                # 模型输入：(w, context) ；输出为0/1，表示context是否为负样本
                w = sentence[i]
                left_context_index = max(0, i - context_size)
                right_context_index = min(len(sentence), i + context_size)
                context = sentence[left_context_index:i] + sentence[i+1:right_context_index+1]
                context += [self.pad] * (2 * context_size - len(context))
                self.data.append((w, context))

        # 负样本数量
        self.n_negatives = n_negatives
        # 负采样分布：若参数ns_dist为None，则使用uniform分布
        self.ns_dist = ns_dist if ns_dist is not None else torch.ones(len(vocab))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):

        # words shape = [batch_size]
        words = torch.tensor([ex[0] for ex in examples], dtype=torch.long)

        # batch_size x context_size * 2 = [16, 4]
        contexts = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        batch_size, context_size = contexts.shape
        neg_contexts = []

        # 对batch内的样本分别进行负采样
        for i in range(batch_size):
            # 保证负样本不包含当前样本中的context
            # 这一步会在负采样分布的context[i]的位置用0来代替
            ns_dist = self.ns_dist.index_fill(0, contexts[i], .0)

            # 输入是ns_dist, 取的次数是 self.n_negatives * context_size，主要是又放回的取样，按照ns_dist的value
            # 进行取样，value越大越容易被选到，这里记录的是下标https://zhuanlan.zhihu.com/p/397976952，

            # 实际上就是对上面contexts[i]不进行采样，因为contexts[i]的位置元素置位0，所以这里权重为0的话，就不会采样了，即负采样
            neg_contexts.append(torch.multinomial(ns_dist, self.n_negatives * context_size, replacement=True))

            # shape=[4 * 10]
            # print(torch.multinomial(ns_dist, self.n_negatives * context_size, replacement=True).shape)

        neg_contexts = torch.stack(neg_contexts, dim=0)

        # 输入中心词[16], 输出背景词[16, 4], 负采样矩阵[16, 40]
        return words, contexts, neg_contexts


class SGNSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SGNSModel, self).__init__()

        # 词嵌入
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 上下文嵌入
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward_w(self, words):
        w_embeds = self.w_embeddings(words)
        return w_embeds

    def forward_c(self, contexts):
        c_embeds = self.c_embeddings(contexts)
        return c_embeds


def get_unigram_distribution(corpus, vocab_size):

    # 从给定语料中统计unigram概率分布, 是计算了一个词频=单词出现次数/所有单词数
    token_counts = torch.tensor([0] * vocab_size)
    total_count = 0
    for sentence in corpus:
        total_count += len(sentence)
        for token in sentence:
            token_counts[token] += 1

    unigram_dist = torch.div(token_counts.float(), total_count)
    return unigram_dist


embedding_dim = 64
context_size = 2
hidden_dim = 16
batch_size = 16
num_epoch = 10
n_negatives = 10

# 读取文本数据
corpus, vocab = load_reuters()
# 计算unigram概率分布
unigram_dist = get_unigram_distribution(corpus, len(vocab))
# 根据unigram分布计算负采样分布: p(w)**0.75
negative_sampling_dist = unigram_dist ** 0.75

# shape= [vocab_size]
negative_sampling_dist /= negative_sampling_dist.sum()

# 构建SGNS训练数据集
dataset = SGNSDataset(corpus, vocab, context_size=context_size, n_negatives=n_negatives, ns_dist=negative_sampling_dist)
data_loader = get_loader(dataset, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SGNSModel(len(vocab), embedding_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
        words, contexts, neg_contexts = [x.to(device) for x in batch]
        optimizer.zero_grad()
        batch_size = words.shape[0]

        # 提取batch内词、上下文以及负样本的向量表示
        # [16, 64, 1]
        word_embeds = model.forward_w(words).unsqueeze(dim=2)

        # [16, 4, 64]
        context_embeds = model.forward_c(contexts)
        neg_context_embeds = model.forward_c(neg_contexts)

        # 正样本的分类（对数）似然 [16, 4]
        context_loss = F.logsigmoid(torch.bmm(context_embeds, word_embeds).squeeze(dim=2))

        # 正样本shape=[16]
        context_loss = context_loss.mean(dim=1)

        # 负样本的分类（对数）似然 [16, 40, 64] * [16, 64, 1] = [16, 40, 1]-->[16, 40]
        neg_context_loss = F.logsigmoid(torch.bmm(neg_context_embeds, word_embeds).squeeze(dim=2).neg())

        # [16, 4, 10] --> [16, 4]
        neg_context_loss = neg_context_loss.view(batch_size, -1, n_negatives).sum(dim=2)

        # shape = [16, 4] --> [16]
        neg_context_loss = neg_context_loss.mean(dim=1)

        # 损失：负对数似然
        loss = -(context_loss + neg_context_loss).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

# 合并词嵌入矩阵与上下文嵌入矩阵，作为最终的预训练词向量
combined_embeds = model.w_embeddings.weight + model.c_embeddings.weight
save_pretrained(vocab, combined_embeds.data, "sgns.vec")

