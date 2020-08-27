import random
import torch
import numpy as np
import torch.nn as nn
torch.random.manual_seed(0)
import torch.nn.functional as F

"""
从 modelA 里面的训练好的参数给相同网络结构的 modelB, 然后把 modelB 进行预训练.
https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/15
"""


def randomSeed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


class ModelA(torch.nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.A = torch.nn.Linear(2, 3)
        self.B = torch.nn.Linear(3, 4)
        self.C = torch.nn.Linear(4, 4)
        self.D = torch.nn.Linear(4, 3)

    def forward(self, x):
        x = F.relu(self.A(x))
        x = F.relu(self.B(x))
        x = F.relu(self.C(x))
        x = F.relu(self.D(x))
        return x


class ModelB(torch.nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.A = torch.nn.Linear(2, 3)
        self.B = torch.nn.Linear(3, 4)
        self.C = torch.nn.Linear(4, 4)
        self.E = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.A(x))
        x = F.relu(self.B(x))
        x = F.relu(self.C(x))
        x = F.relu(self.E(x))
        return x


randomSeed(0)
modelA = ModelA()
modelA_dict = modelA.state_dict()
print('='*40)
for key in sorted(modelA_dict.keys()):
    parameter = modelA_dict[key]
    print('the param name is:\n', key)
    print('the param size is:\n', parameter.size())
    print('the param value is:\n', parameter)
    print("="*40)

modelB = ModelB()
modelB_dict = modelB.state_dict()
for key in sorted(modelB_dict.keys()):
    parameter = modelB_dict[key]
    print('the param name is:\n', key)
    print('the param size is:\n', parameter.size())
    print('the param value is:\n', parameter)
    print('='*40)
print("modelB is going to use the ABC layers parameters from modelA")

pretrained_dict = modelA_dict
model_dict = modelB_dict

# 1. filter out unnecessary keys
newdict = dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
print('============== the predict ============')
print(pretrained_dict)

# 2. overwrite entries in the existing state dict, put the modelA param to modelB
model_dict.update(pretrained_dict)

# 3. load the new state dict
modelB.load_state_dict(model_dict)
modelB_dict = modelB.state_dict()
print("\n=============== after convert the modelA value to modelB ===========")
for key in sorted(modelB_dict.keys()):
    parameter = modelB_dict[key]
    print(key)
    print(parameter.size())
    print(parameter)
    print("="*40)

for k, v in dict(modelB.named_parameters()).items():
    print('the k is:\n', k)
    print("the v is:\n", v)
