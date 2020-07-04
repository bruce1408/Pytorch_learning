import torch
import torch.utils.data as data
import torchvision.transforms as transforms

train_x = [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
           torch.tensor([1, 2, 3, 4, 5, 6, 7]),
           torch.tensor([2, 3, 4, 5, 6, 7]),
           torch.tensor([3, 4, 5, 6, 7]),
           torch.tensor([4, 5, 6, 7]),
           torch.tensor([5, 6, 7]),
           torch.tensor([6, 7]),
           torch.tensor([7])]


def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    train_data = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data.unsqueeze(-1), data_length  # 对train_data增加了一维数据

    # return train_data, data_length


class MyData(data.Dataset):
    def __init__(self, train_x):
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, item):
        return self.train_x[item]


# train_data, data_length = collate_fn(train_x)
# print(train_data.shape)
# print(data_length)
# train_data = MyData(train_x)
train_dataloader = data.DataLoader(train_x, batch_size=2, collate_fn=collate_fn)

for data, length in train_dataloader:
    print(data, end='')
    print(length)
    print(torch.nn.utils.rnn.pack_padded_sequence(data, length, batch_first=True))
net = torch.nn.LSTM(1, 5, batch_first=True)
