import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class Net(nn.Module):
    """
    DSSM
    """
    def __init__(self, embed_size, label_size, dropout=0.2):
        super(Net, self).__init__()
        self.embed = nn.Embedding(embed_size, 100)
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, label_size)
        self.dropout = nn.Dropout(dropout)
        # self.Sigmoid = nn.Sigmoid() # method1
        self.relu = nn.ReLU()
        self.name = "DSSM"

    def forward(self, a, b, lengths_a=None, lengths_b=None):

        a = self.embed(a).sum(1)
        b = self.embed(b).sum(1)

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
    model = Net(5, 3)
    output = model(input1, input2)
    print(output.shape, output)

    import os
    import torch.optim as optim

    file = "/Users/bruce/Downloads/requirements.txt"

    # with open(file, "r") as f:
    #     txt = f.read()
    #     for eachlein in f:
    #         print(eachlein)
    # print(txt)
    Cosine_lr = True
    if True:
        lr = 1e-4
        Batch_size = 4
        Freeze_Epoch = 10
        Unfreeze_Epoch = 20  # 总共 100 epoch
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        for epoch in range(200):
            lr_scheduler.step()
            print("epoch: %d, lr:%f"%(epoch, lr_scheduler.get_lr()[0]))