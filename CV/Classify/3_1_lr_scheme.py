# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
#
# model = nn.Conv2d(3, 64, 3)
# optimizer = optim.SGD(model.parameters(), lr=0.5)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)
# lrs = []
# for i in range(5):
#     optimizer.zero_grad()
#     x = model(torch.randn(3, 3, 64, 64))
#     loss = x.sum()
#     loss.backward()
#     lrs.append((i, optimizer.param_groups[0]['lr']))
#     optimizer.step()
#     lr_scheduler.step()
#
# lr = np.array(lrs)
#
# # 绘制loss 变化情况
# plt.plot(lr[:, 0], lr[:, 1])
# plt.show()
#
# optimizer_state = optimizer.state_dict()
# scheduler_state = lr_scheduler.state_dict()
#
# # resume
# optimizer = optim.SGD(model.parameters(), lr=0.5)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)
# optimizer.load_state_dict(optimizer_state)  # 只加载optimizer的状态
#
# for i in range(5):
#     optimizer.zero_grad()
#     x = model(torch.randn(3, 3, 64, 64))
#     loss = x.sum()
#     loss.backward()
#     print('{} optim: {}'.format(i, optimizer.param_groups[0]['lr']))
#     optimizer.step()
#     print('{} scheduler: {}'.format(i, lr_scheduler.get_last_lr()))
#     lr_scheduler.step()


import torch
import torch.nn as nn
import torch.optim as optim

# scheme的时候会改变lr，同时更新optimizer里面的lr，所以应该先optimizer，然后再scheduler
model = nn.Conv2d(3, 64, 3)
optimizer = optim.SGD(model.parameters(), lr=0.5)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)

# 0 optim: 0.5
# 1 scheduler: 0.5
# 1 optim: 0.05
# 2 scheduler: 0.005000000000000001
# 2 optim: 0.05
# 3 scheduler: 0.05
# 3 optim: 0.005000000000000001
# 4 scheduler: 0.0005000000000000001
# 4 optim: 0.005000000000000001
for i in range(5):
    optimizer.zero_grad()
    print('{} scheduler: {}'.format(i, lr_scheduler.get_lr()[0]))
    # lr_scheduler.step()
    x = model(torch.randn(3, 3, 64, 64))
    loss = x.sum()
    loss.backward()
    print('{} optim: {}'.format(i, optimizer.param_groups[0]['lr']))
    optimizer.step()
    lr_scheduler.step()

