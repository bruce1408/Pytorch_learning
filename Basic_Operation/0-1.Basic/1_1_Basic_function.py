# coding=utf-8
import torch
import random
from torch import nn
import numpy as np

SEED = 1234


def randomSeed(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


randomSeed(SEED)
# hyper parameters
in_dim = 1
n_hidden_1 = 1
n_hidden_2 = 1
out_dim = 1
"""
module 和 children区别
module 是深度优先遍历打印出网络结构,而 children是只打印出网络的子结构,不再管子结构的下一结构
"""


class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
        )
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

        print("children")
        for i, module in enumerate(self.children()):
            print(i, module)

        print("modules")
        for i, module in enumerate(self.modules()):
            print(i, module)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = Net(in_dim, n_hidden_1, n_hidden_2, out_dim)
"""
torch.eq函数
"""
a = torch.FloatTensor([1, 2, 3])
b = torch.FloatTensor([2, 21, 3])
print(b.eq(a.data).cpu())
# output is [false, false, True]

"""
embedding vector
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
print("="*50)
word_to_ix = {'hello': 1, 'world': 2}
embeds = nn.Embedding(7, 5, padding_idx=0)
hello_idx = torch.LongTensor([word_to_ix['hello']])
hello_embed = embeds(hello_idx)
print(hello_embed)
print(hello_embed.shape)
print(embeds(torch.LongTensor([1, 4, 3])))
print("the two shape is :\n",embeds(torch.LongTensor([[1, 4, 3], [2, 3, 1]])).shape)

inputs = torch.randint(1, 7, (3, 3))
print(embeds(inputs).shape)
print(embeds(inputs))

weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)
print(embedding)
input = torch.LongTensor([1])
print(embedding(input))

"""
torch.mm  矩阵乘法
torch.bmm 三维矩阵乘法，第一维是batchsize
torch.matmul 广播乘法,1维的话返回标量,2维的话返回矩阵乘法结果,其他情况比较复杂,具有广播功能
torch.mul 对位相乘,两个矩阵的维度必须一致才可以
torch.dot
"""
import torch

a1 = torch.Tensor([1, 2, 3])
a2 = torch.Tensor([1, 2, 3])
a = torch.Tensor([[1, 2], [2, 3]])
b = torch.Tensor([[2, 3], [2, 3]])
c = torch.mul(a, b)
print('对位相乘, a 和 b 维度必须相同， mul is:\n', c)
d = torch.mm(a, b)
print("矩阵乘法 mm is:\n", d)
e = torch.matmul(a, b)
print("广播乘法 matmul is:\n", e)
f = torch.dot(a1, a2)
print('一维点乘 dot is:\n', f)
print("=" * 50)
"""
print 格式化输出
"""
# import math
# print('{0:n}'.format(20.000))
# train_loss = 0.8797
# print(f'Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')  # 7 前面是空格
# print('Train loss: {:.3f} | train ppl: {:7.3f}'.format(train_loss, math.exp(train_loss)))

"""
masked_fill， 使用mask 技巧
"""
# import torch.nn.functional as F
# import numpy as np
#
# a = torch.Tensor([1, 2, 3, 4])
# a = a.masked_fill(mask=torch.ByteTensor([1, 1, 0, 0]), value=-np.inf)
# print(a)
# b = F.softmax(a, dim=0)
# print(b)
# print(-np.inf)
# print("#"*10)
# a = torch.tensor([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7]], [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]])
# print(a)
# print(a.size())
# mask = torch.ByteTensor([[[0]], [[1]]])
# print(mask.size())
# b = a.masked_fill(mask, value=torch.tensor(-1e9))
# print(b)
# print(b.size())
# print("#"*10)
#
# a = torch.tensor([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7]], [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]])
# print(a)
# print(a.size())
# mask = torch.ByteTensor([[[1], [1], [0]], [[0], [1], [1]]])
# print(mask.size())
# b = a.masked_fill(mask, value=torch.tensor(-1e9))
"""
tensor b is:
tensor([[[-1000000000, -1000000000, -1000000000, -1000000000],
         [-1000000000, -1000000000, -1000000000, -1000000000],
         [          7,           7,           7,           7]],

        [[          1,           1,           1,           1],
         [-1000000000, -1000000000, -1000000000, -1000000000],
         [-1000000000, -1000000000, -1000000000, -1000000000]]])
"""
# c = a.masked_fill(mask == 0, value=torch.tensor(-1e9))  # mask等于0的对应的行列元素赋值为value
"""
tensor c is:
tensor([[[          5,           5,           5,           5],
         [          6,           6,           6,           6],
         [-1000000000, -1000000000, -1000000000, -1000000000]],

        [[-1000000000, -1000000000, -1000000000, -1000000000],
         [          2,           2,           2,           2],
         [          3,           3,           3,           3]]])
"""

'''转换向量过程'''
"""
repeat 和 expand 两个函数的区别, 不会分配新内存,只是创建一个新的视图, 且只能扩展维度是1的张量
"""
# a = torch.tensor([1, 2, 3, 4])
# a1 = a.expand(8, 4)  # 不会分配新内存,只是创建一个新的视图,且只能扩展维度是1的张量
# print(a1.shape)  # [8, 4]
# b1 = a.repeat(3, 2)  # 沿着特定维度重复张量
# print(b1.shape)  # shape =[3, 8]

"""
meshgrid函数使用,构成一个坐标系可用,选择两个行列坐标的值,然后最后生成的坐标
x和y轴的参数mn分别是较小的值作为行,大的作为列
"""
# x = np.linspace(0, 1, 5)
# y = np.linspace(0, 1, 3)
# xc, yc = np.meshgrid(x, y)
# print(xc)
# print(yc)
# xc = xc.reshape(-1, 1)
# yc = yc.reshape(-1, 1)
# c = np.concatenate((xc, yc), axis=1)
# print(c)

"""
torch.floor不超过这个数的最大整数
tensor([[ 0.0461,  0.4024, -1.0115],
        [ 0.2167, -0.6123,  0.5036]])
tensor([[ 0.,  0., -2.],
        [ 0., -1.,  0.]])
"""
# a = torch.randn((2, 3))
# b = torch.floor(a)
# print(a)
# print(b)
"""
上采样函数
nn.ConvTranspose2d,有参数可以训练
hout = (hin-1)*stride - 2 * padding + kernel + output_padding
参考链接https://blog.csdn.net/qq_27261889/article/details/86304061

nn.Unsample 上采样没有参数,速度更快,采样策略给定

参考链接 
https://www.shuzhiduo.com/A/gGdX9OPWz4/
https://blog.csdn.net/wangweiwells/article/details/101820932
https://zhuanlan.zhihu.com/p/87572724(详解 align_corners=False, True用法)
双线性插值算法
https://juejin.im/post/6844903924999127047
https://zhuanlan.zhihu.com/p/110754637
"""
# input = torch.ones((2, 2, 3, 4))
# output = nn.ConvTranspose2d(2, 4, kernel_size=4, stride=1, padding=0, bias=False)
# # [2, 4, 6, 7]
# print(output(input).shape)
#
# input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
# m = nn.Upsample(scale_factor=2, mode="nearest")
# output = m(input)
# print(output)
# m = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
# output = m(input)

"""
使用交叉熵函数
"""
# import torch
# import torch.nn as nn
# import math
#
# entroy=nn.CrossEntropyLoss()
# input=torch.Tensor([[-0.7715, -0.6205, -0.2562]])
# target = torch.tensor([0])
#
# output = entroy(input, target)
# print(output)
