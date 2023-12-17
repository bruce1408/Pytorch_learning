import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Normalize, ToTensor, Compose
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

# 用 mnist 作为 toy dataset
mnist = torchvision.datasets.MNIST(root='mnist', download=True, transform=ToTensor())
dataloader = DataLoader(dataset=mnist, batch_size=8)

# 初始化一个带 BN 的简单模型
toy_model = nn.Sequential(
    nn.Linear(28 ** 2, 128), 
    nn.BatchNorm1d(128),
    nn.ReLU(), 
    nn.Linear(128, 10), 
    nn.Sigmoid())

optimizer = torch.optim.SGD(toy_model.parameters(), lr=0.1)

bn_1d_layer = toy_model[1]
# print the length 128 of 4
print(f'Initial weight is {bn_1d_layer.weight[:4].tolist()}...')
print(f'Initial bias is {bn_1d_layer.bias[:4].tolist()}...\n')
# 模拟更新2次参数
for (i, data) in enumerate(dataloader):
    print(data[0].shape)
    print(data[1].shape)
    output = toy_model(data[0].view(data[0].shape[0], -1))
    (F.cross_entropy(output, data[1])).backward()
    # 输出部分参数的梯度，验证weight和bias确实是通过gradient descent更新的
    print(f'Gradient of weight is {bn_1d_layer.weight.grad[:4].tolist()}...')
    print(f'Gradient of bias is {bn_1d_layer.bias.grad[:4].tolist()}...')
    optimizer.step()
    optimizer.zero_grad()
    if i == 1:
        break
print(f'\nNow weight is {bn_1d_layer.weight[:4].tolist()}...')
print(f'Now bias is {bn_1d_layer.bias[:4].tolist()}...')

inputs = torch.randn(4, 128)
bn_outputs = bn_1d_layer(inputs)
new_bn = nn.BatchNorm1d(128)
bn_outputs_no_weight_bias = new_bn(inputs)

print(bn_1d_layer.weight[0:4])
print(new_bn.weight[0:4])

assert not torch.allclose(bn_outputs, bn_outputs_no_weight_bias)

print(bn_outputs[0:1])
print(bn_outputs_no_weight_bias[0:1])