import torch
lr = 0.001
gamma = 0.96
from torchvision.models import resnet50
model = resnet50(pretrained=True)
"""
调整学习率,一个是指数衰减, 还有一个是使用自定义规则的lambda衰减.
衰减公式为 lr = lr * (gamma ** epoch)
https://www.cnblogs.com/wanghui-garcia/p/10895397.html
"""
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
rule = lambda epoch: 0.96**epoch
lrr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
lre = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)


for epoch in range(40):
    print(epoch, lrr.get_lr()[0])
    lrr.step()
    lre.step()