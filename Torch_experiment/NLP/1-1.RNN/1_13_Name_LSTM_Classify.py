import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
# sys.path.append("../")
# from tools.name_data import NameDataset
# from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence


# 这个代码和1_10_Name_GRU_Classify.py同一个功能，但是使用的是lstm函数搭建的网络结构
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Parameters and DataLoaders
HIDDEN_SIZE = 100
N_LAYERS = 2
BATCH_SIZE = 256
N_EPOCHS = 30
N_CHARS = 128  # ASCII

# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import csv
import gzip


def str2ascii_arr(msg):
    arr = [ord(c) for c in msg]
    return arr, len(arr)


class NameDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self, is_train_set=False):
        filename = '../../data/names_train.csv.gz' if is_train_set else '../../data/names_test.csv.gz'
        with gzip.open(filename, "rt") as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.names = [row[0] for row in rows]
        self.countries = [row[1] for row in rows]
        self.len = len(self.countries)
        print(self.len)

        self.country_list = list(sorted(set(self.countries)))

    def __getitem__(self, index):
        return str2ascii_arr(self.names[index]), self.countries[index]

    def __len__(self):
        return self.len

    def get_countries(self):
        return self.country_list

    def get_country(self, id):
        return self.country_list[id]

    def get_country_id(self, country):
        return self.country_list.index(country)


def collate_fn(data):

    # 输入，是一个list，每个list内部数字是一个tensor
    input_name = [torch.tensor(eachline[0][0]) for eachline in data]

    # label一般用long tensor类型
    country = torch.LongTensor([train_dataset.get_country_id(eachline[1]) for eachline in data])

    # length是一个tensor数组
    length = torch.tensor([eachline[0][1] for eachline in data])

    # input_name进行pad处理
    input_name = pad_sequence(input_name, batch_first=True, padding_value=0)

    return create_variable(input_name), create_variable(length), create_variable(country)


test_dataset = NameDataset(is_train_set=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


train_dataset = NameDataset(is_train_set=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

N_COUNTRIES = len(train_dataset.get_countries())
print(N_COUNTRIES, "countries")


# Some utility functions
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


def countries2tensor(countries):
    country_ids = [train_dataset.get_country_id(country) for country in countries]
    return torch.LongTensor(country_ids)


class RNNClassifier(nn.Module):

    # Our model
    def __init__(self, vocab_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.n_layers,
                            bidirectional=bidirectional, batch_first=True)

        self.fc1 = nn.Linear(hidden_size * self.n_directions * self.n_layers, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, seq_lengths):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: B x S (input size), transpose to make S(sequence) x B (batch)
        # input = input.t()
        batch_size = input.size(0)

        # Make a hidden, hidden = self._init_hidden(batch_size)

        # Embedding B x S-> B x S x I (embedding size)
        embedded = self.embedding(input)

        # Pack them up nicely
        gru_input = pack_padded_sequence(embedded, seq_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)

        # To compact weights again call flatten_parameters().
        # self.lstm.flatten_parameters()
        output, (hidden, ct) = self.lstm(gru_input)
        print(hidden.shape)

        # 如果是状态h或者是c，不能用pad_packed_sequence函数来进行填充0；但是输出output可以进行填充，不知道原因
        # hidden, _ = pad_packed_sequence(hidden, batch_first=True)

        # Use the last layer output as FC's input No need to unpack, since we are going to use hidden
        # 对输出结果hidden先进行一个转换，[batch x n-layers*bidireciton x hidden-size]--> [batch, n-layers*bidireciton x hidden-size]
        hidden = hidden.permute(1, 0, 2).reshape(batch_size, -1)

        # 接一个全连接层
        fc_output = self.fc1(hidden)

        # 进行softmax函数进行计算
        out = self.softmax(fc_output)
        return out

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_variable(hidden)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Train cycle
def train():
    total_loss = 0

    for i, (input, seq_lengths, target) in enumerate(train_loader, 1):
        # input, seq_lengths, target = make_variables(names, countries)
        # input = pack_padded_sequence(input, seq_lengths, batch_first=True, enforce_sorted=False)

        # print(input.shape, target.shape, seq_lengths.shape)
        # print(input)
        output = classifier(input, seq_lengths)

        loss = criterion(output, target)
        total_loss += loss.item()

        classifier.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.2f}'.format(
                time_since(start), epoch,  i * input.shape[0], len(train_loader.dataset),
                100. * i * input.shape[0] / len(train_loader.dataset), total_loss / i * input.shape[0]))

    return total_loss


# Testing cycle
def test(name=None):
    # Predict for a given name
    if name:
        # input, seq_lengths, target = make_variables([name], [])
        sequence_and_length = [str2ascii_arr(word) for word in name]
        vectorized_seqs = [torch.tensor(sl[0]) for sl in sequence_and_length]
        seq_lengths = create_variable(torch.LongTensor([sl[1] for sl in sequence_and_length]))

        test_input = create_variable(pad_sequence(vectorized_seqs, batch_first=True))

        output = classifier(test_input, seq_lengths)
        pred = output.max(1, keepdim=True)[1]
        country_id = pred.cpu().numpy()[0][0]
        print(name, "is", train_dataset.get_country(country_id))
        return

    print("evaluating trained model ...")
    correct = 0
    test_data_size = len(test_loader.dataset)

    for batch in (test_loader):
        # 这里有两种转化方式，一种是使用to(device)，或者是使用create_variable函数
        inputs, seq_lengths, targets = [x.to(device) for x in batch]

        output = classifier(inputs, seq_lengths)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(targets.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, test_data_size, 100. * correct / test_data_size))


if __name__ == '__main__':

    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRIES, N_LAYERS)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
        classifier = nn.DataParallel(classifier)

    if torch.cuda.is_available():
        classifier.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        train()

        # Testing
        test()

        # Testing several samples
        test("Sung")
        test("Jungwoo")
        test("Soojin")
        test("Nako")