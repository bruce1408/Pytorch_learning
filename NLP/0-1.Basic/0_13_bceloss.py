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
# #根据公式计算


import torch
import torch.nn as nn
import math

criterion = nn.CrossEntropyLoss()
output = torch.randn(1, 5, requires_grad=True)
label = torch.empty(1, dtype=torch.long).random_(5)
loss = criterion(output, label)

print("网络输出为5类:")
print(output)
print("要计算label的类别:")
print(label)
print("计算loss的结果:")
print(loss)

first = 0
for i in range(1):
    first = -output[i][label[i]]
second = 0
for i in range(1):
    for j in range(5):
        second += math.exp(output[i][j])
res = 0
res = (first + math.log(second))
print("自己的计算结果：")
print(res)

