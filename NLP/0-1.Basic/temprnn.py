from torchvision.models.resnet import resnet18
import torch.nn as nn
import torch

model = resnet18(pretrained=True)
print(model)
print('='*50)
print(list(model.children())[:-1])
print("="*60)
# for param in model.parameters():
#     print(type(param.data), param.size())
#     print(list(param.data))
print(model.state_dict().keys())
print('='*90)
for key in model.state_dict():
    print(key, "corr= ", model.state_dict()[key])