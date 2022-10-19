import os
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torchsummary import summary
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
"""
pytorch 官方实现分类模型代码汇总
https://pytorch.org/docs/stable/torchvision/models.html
https://github.com/weiaicunzai/pytorch-cifar100/tree/master/models
"""


class AlexNet(nn.Module):
    def __init__(self, n_class):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, n_class)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


if __name__ == '__main__':
    net = AlexNet(n_class=2)
    if torch.cuda.is_available():
        summary(net.cuda(), (3, 224, 224))
        # input [batch_size, channel, width, heigth]
        x = torch.rand(size=(8, 3, 224, 224)).to('cuda')
        output = net(x)
        print('the output shape is: ', output.size())
    else:
        summary(net, (3, 224, 224))

