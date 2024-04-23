import math
import torch
import numpy as np
import random
import torch.nn as nn
"""
参考资料
https://www.cnblogs.com/marsggbo/p/10401215.html
https://blog.csdn.net/b1055077005/article/details/100152102
"""
SEED = 1234


def randomSeed(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


randomSeed(SEED)


def crossentory():
    criterion = nn.CrossEntropyLoss()
    output = torch.randn(1, 5, requires_grad=True)
    label = torch.empty(1, dtype=torch.long).random_(5)
    loss = criterion(output, label)

    print("{:=^50s}".format("网络输出为5类"))
    print(output)

    print('{:=^50s}'.format("要计算label的类别"))
    print(label)

    print("{:=^50s}".format("计算loss的结果"))
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


crossentory()