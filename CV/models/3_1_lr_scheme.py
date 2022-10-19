import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# 绘制loss 变化情况
def plot_lr(lr, title, step=10):
    plt.plot(lr[:, 0], lr[:, 1])
    plt.tick_params(axis='both', labelsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xlabel('%s' % title, fontsize=14)
    plt.axis([0, step, 0, 1])
    for a, b in zip(lr[:, 0], lr[:, 1]):
        plt.text(a, b + 0.1, '%.5f' % b, ha='left', va='bottom', rotation=45, fontsize=8)
    plt.show()
    plt.close()


model = nn.Conv2d(3, 64, 3)
optimizer = optim.SGD(model.parameters(), lr=0.5)


# # scheme的时候会改变lr，同时更新optimizer里面的lr，所以应该先optimizer，然后再scheduler
def train_model(lr_policy):
    lrs = []
    for i in range(20):
        optimizer.zero_grad()
        x = model(torch.randn(3, 3, 64, 64))
        loss = x.sum()
        loss.backward()
        # print((optimizer.param_groups))
        lrs.append((i, optimizer.param_groups[0]['lr']))
        optimizer.step()
        lr_policy.step()
        print(optimizer.param_groups[0]['lr'])
    lr = np.array(lrs)
    return lr


# LR 等间隔调整学习率
# 调整倍数gamma倍，每隔step步进行调整，默认是0.1倍下降
# step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)
# steplr = train_model(step_lr_scheduler)
# plot_lr(steplr, "stepLR size")

# multiStepLR 指定step进行学习率的调整
# multi_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6], gamma=0.1)
# multisteplr = train_model(multi_lr_scheduler)
# plot_lr(multisteplr, "multistepLR size")


# exponentialLR 按照指数衰减进行调整学习率
# lr = lr * gamma**epoch
# exponential_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)
# exponeLR = train_model(exponential_lr_scheduler)
# plot_lr(exponeLR, "exponeLR")


# cosineAnnealingLR
# T_max表示周期，在T_max周期之后重新按照设定的学习率进行，然后eta_min表示最小的学习率
# cosine_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.003)
# cosineLR = train_model(cosine_lr_scheduler)
# plot_lr(cosineLR, "cosineLR", 30)


# warmup 学习率
"""
Implements gradual warmup, if train_steps < warmup_steps, the
learning rate will be `train_steps/warmup_steps * init_lr`.
Args:
    warmup_steps:warmup步长阈值,即train_steps<warmup_steps,使用预热学习率,否则使用预设值学习率
    train_steps:训练了的步长数
    init_lr:预设置学习率
"""
warmup_steps = 2500
init_lr = 0.1
# 模拟训练15000步
max_steps = 15000
for train_steps in range(max_steps):
    if warmup_steps and train_steps < warmup_steps:
        warmup_percent_done = train_steps / warmup_steps
        warmup_learning_rate = init_lr * warmup_percent_done  # gradual warmup_lr
        learning_rate = warmup_learning_rate
    else:
        # learning_rate = np.sin(learning_rate)  #预热学习率结束后,学习率呈sin衰减
        learning_rate = learning_rate ** 1.0001  # 预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)
    if (train_steps + 1) % 100 == 0:
        print("train_steps:%.3f--warmup_steps:%.3f--learning_rate:%.3f" % (
            train_steps + 1, warmup_steps, learning_rate))
