import sys
import torch
import jieba

sys.path.append("../")
from itertools import chain
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from utils.json_extra import read_json
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict, Counter


class BertData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):  # 返回数据长度
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind]


class CustomData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    lengths_first = torch.tensor([len(ex[0]) for ex in examples])
    lengths_second = torch.tensor([len(ex[1]) for ex in examples])
    first_sen = [torch.tensor(ex[0]) for ex in examples]
    second_sen = [torch.tensor(ex[1]) for ex in examples]
    labels = torch.LongTensor([int(ex[2]) for ex in examples])

    first_sen = pad_sequence(first_sen, batch_first=True, padding_value=0)
    second_sen = pad_sequence(second_sen, batch_first=True, padding_value=0)
    return first_sen, second_sen, lengths_first, lengths_second, labels


def collate_fn_test(examples):
    lengths_first = torch.tensor([len(ex[0]) for ex in examples])
    lengths_second = torch.tensor([len(ex[1]) for ex in examples])
    first_sen = [torch.tensor(ex[0]) for ex in examples]
    second_sen = [torch.tensor(ex[1]) for ex in examples]

    first_sen = pad_sequence(first_sen, batch_first=True, padding_value=0)
    second_sen = pad_sequence(second_sen, batch_first=True, padding_value=0)
    return first_sen, second_sen, lengths_first, lengths_second


def collate_fn_bert(examples):
    first_setence = [ex[0] for ex in examples]
    second_setence = [ex[1] for ex in examples]
    labels = torch.LongTensor([int(ex[2]) for ex in examples])
    return first_setence, second_setence, labels


def collate_fn_bert_test(examples):
    first_setence = [ex[0] for ex in examples]
    second_setence = [ex[1] for ex in examples]
    # labels = torch.LongTensor([int(ex[2]) for ex in examples])
    return first_setence, second_setence


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


def cut_sentence(path, train=True, bertModel=False):
    # 第一句放一个集合，第二句放一个集合
    data = read_json(path)
    if bertModel:
        total_data = [(eachpair[1], eachpair[2], int(float(eachpair[3]))) \
                          if train else (eachpair[1], eachpair[2]) for eachpair in data]
    else:
        total_data = [
            (jieba.lcut(eachpair[1], cut_all=False), jieba.lcut(eachpair[2], cut_all=False), int(float(eachpair[3]))) \
                if train else (jieba.lcut(eachpair[1], cut_all=False), jieba.lcut(eachpair[2], cut_all=False)) for
            eachpair in data]

    return total_data


def generate_vocab(sentences):
    # total = [word for sentence in sentences for word in sentence]
    vocab = Vocab(sentences)
    return vocab


def save_vocab(vocab, path):
    with open(path, 'w') as writer:
        writer.write("\n".join(vocab.idx_to_token))


def read_vocab(path):
    with open(path, 'r') as f:
        tokens = f.read().split('\n')
    return Vocab(tokens)


def generate_data(vocab, train_data, val_data):
    train_data = [(vocab.convert_tokens_to_ids(pairdata[0]), vocab.convert_tokens_to_ids(pairdata[1]), pairdata[2]) for
                  pairdata in train_data]
    val_data = [(vocab.convert_tokens_to_ids(pairdata[0]), vocab.convert_tokens_to_ids(pairdata[1]), pairdata[2]) for
                pairdata in val_data]
    return train_data, val_data


# Constants
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
BOW_TOKEN = "<bow>"
EOW_TOKEN = "<eow>"

if __name__ == "__main__":
    # 生成单词词典
    path_train = "../data/KUAKE-QQR_train.json"
    path_test = "../data/KUAKE-QQR_dev.json"
    train_data = cut_sentence(path_train, bertModel=True)
    val_data = cut_sentence(path_test)
    print(train_data, train_data.__len__())
    # 生成词典的过程
    # total_sentence = [word for each_pair in train_data for word in each_pair[0]]
    # total_sentence += [word for each_pair in train_data for word in each_pair[1]]
    # total_sentence += [word for each_pair in val_data for word in each_pair[0]]
    # total_sentence += [word for each_pair in val_data for word in each_pair[1]]
    # # print(total_sentence)
    # vocab = generate_vocab(total_sentence)

    # # print the len of the vocab
    # print(len(vocab))
    # save_vocab(vocab, "./data/vocab")
    # 保存词典

    # 字符级别
    # txt = Vocab.build(sen, min_freq=2, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    # print(txt.idx_to_token)

    # # 生成词典的过程
    # total_sentence = [word for each_pair in train_data for word in each_pair[0]]
    # total_sentence += [word for each_pair in train_data for word in each_pair[1]]
    # total_sentence += [word for each_pair in val_data for word in each_pair[0]]
    # total_sentence += [word for each_pair in val_data for word in each_pair[1]]
    # vocab = generate_vocab(total_sentence)
    #
    # # # print the len of the vocab
    # print(len(vocab))
    #
    # train_data, val_data = generate_data(vocab, train_data, val_data)
    # print(train_data.__len__(), val_data.__len__())
    #
    # train_dataset = CustomData(train_data)
    # train_data_loader = DataLoader(train_dataset, batch_size=3, collate_fn=collate_fn, shuffle=True)
    #
    # epoch = 64
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
    #     first_txt, second_txt, lengths_fir, lengths_second, labels = [x.to(device) for x in batch]
    #     print(first_txt.shape, second_txt.shape, labels.shape, lengths_fir.shape)

    # sen = ["hello", "my", "name", "is", "cui", "dong", "dong"]
    # txt = Vocab(sen)
