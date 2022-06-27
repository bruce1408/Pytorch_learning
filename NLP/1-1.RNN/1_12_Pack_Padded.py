import torch
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

"""
参考资料:
https://www.cnblogs.com/sbj123456789/p/9834018.html
对于序列长度可变的情况，介绍使用pad_sequence函数的用法
对于序列不等长的情况，使用填充0来进行等长序列的操作，0的填充可能需要知道所有数据的最大长度，然后开始填充，这样不是很合理，因为
按照批次进行的话，我们希望得到每个批次里面最大的长度即可，然后开始进行填充。使用pad_sequence 函数里面的collate_fn函数来进行操作即可
"""
batch_size = 2
train_x = [torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
           torch.FloatTensor([1, 2, 3, 4, 5, 6, 7]),
           torch.FloatTensor([2, 3, 4, 5, 6, 7]),
           torch.FloatTensor([3, 4, 5, 6, 7]),
           torch.FloatTensor([4, 5, 6, 7]),
           torch.FloatTensor([5, 6, 7]),
           torch.FloatTensor([6, 7]),
           torch.FloatTensor([7])]  # 数据类型是浮点型


class MyData(data.Dataset):
    """
    这里什么都不做，把数据补0功能放到collate_fn这个函数里面去
    """
    def __init__(self, train_x):
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, item):
        return self.train_x[item]


def collate_fn(train_data):
    """
    该函数的功能就是对train_data数据进行填充，填充原则是对当前批次的数据长度先要进行排序
    按照从大到小的顺序排序，然后开始填充
    :param train_data:
    :return:
    """

    # 按照长度排序
    train_data.sort(key=lambda data: len(data), reverse=True)

    # 得到排序后的数据的长度列表
    data_length = [len(data) for data in train_data]

    # 对该数据进行填充
    train_data = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True, padding_value=0)

    print('train data shape: ', train_data.shape)
    # 对train_data最后增加了一维数据，返回[batch_size, seq_len, input_size]数据和长度
    return train_data.unsqueeze(-1), data_length


# 定义网络
net = torch.nn.LSTM(input_size=1, hidden_size=5, num_layers=1, batch_first=True)


train_data = MyData(train_x)

train_dataloader = data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)  # 进行数据处理
for data, length in train_dataloader:
    # 对于之前的加0操作的数据进行压缩，然后直接丢给lstm进行运算
    print(data.shape)
    data_ = torch.nn.utils.rnn.pack_padded_sequence(data, length, batch_first=True)
    print('the data_ is: \n', data_)
    output, (ht, ct) = net(data_)
    print("output is: ", output)
    # 预算结果再解压缩补0然后提取结果
    output_, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
    print(output_.shape)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 并行gpu
        torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
        torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速




def blstm():
    lstm = torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=4, batch_first=True, bidirectional=True)
    x = torch.rand(2, 6, 3)
    hidden_state = torch.zeros(4 * 2, 2, 5)
    cell_state = torch.zeros(8, 2, 5)
    outputs, (hidden_state, cell_state) = lstm(x, (hidden_state, cell_state))

    print(outputs.shape)
    print(hidden_state[-1, :, :].shape)
    print(cell_state.shape)
    # print(outputs)


if __name__=="__main__":
    set_seed(0)
    blstm()
