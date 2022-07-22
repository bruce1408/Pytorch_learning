import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class DSSM(nn.Module):

    def __init__(self, embed_size, label_size, dropout=0.2):
        super(DSSM, self).__init__()
        self.embed = nn.Embedding(embed_size, 100)
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, label_size)
        self.dropout = nn.Dropout(dropout)
        # self.Sigmoid = nn.Sigmoid() # method1
        self.relu = nn.ReLU()

    def forward(self, a, b, lengths_a, lengths_b):

        a = self.embed(a).sum(1)
        b = self.embed(b).sum(1)
        # a = self.embed(a)
        # b = self.embed(b)

        # a = pack_padded_sequence(a, lengths_a, batch_first=True, enforce_sorted=False)
        # b = pack_padded_sequence(b, lengths_b, batch_first=True, enforce_sorted=False)


        # torch.tanh
        a = self.relu(self.fc1(a))
        # print('fc1', a.shape)
        # print('relu', a)
        # a = self.dropout(a)
        a = self.relu(self.fc2(a))
        # a = self.dropout(a)
        a = self.relu(self.fc3(a))
        # a = self.dropout(a)

        b = self.relu(self.fc1(b))
        # b = self.dropout(b)
        b = self.relu(self.fc2(b))
        # b = self.dropout(b)
        b = self.relu(self.fc3(b))
        output = torch.abs(a - b)
        output1 = torch.mul(a, b)
        res = output + output1
        # print('res shape ', res.shape)
        # res = torch.cat((output1, output), 1)
        # print(res.shape)
        # b = self.dropout(b)
        # print(a.shape, b.shape)
        # cosine = torch.cosine_similarity(a, b, dim=1)  # 计算两个句子的余弦相似度
        # print(cosine.shape)
        # cosine = self.Sigmoid(cosine-0.5)
        # cosine = self.relu(cosine)
        # cosine = torch.clamp(cosine, 0, 1)
        return res


if __name__ == "__main__":
    input1 = torch.randint(0, 5, (2, 5))
    input2 = torch.randint(0, 5, (2, 5))
    model = DSSM(5)
    output = model(input1, input2)
    print(output.shape, output)