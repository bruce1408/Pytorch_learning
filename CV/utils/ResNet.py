import torch
import torch.nn as nn
from torchsummary import summary
"""
实现pytorch ResNet50网络结构
"""


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsampling=False, dim=4):
        super(Bottleneck, self).__init__()
        self.dim = dim
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(
            # 卷积层1
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),  # 默认padding=0
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            # 卷积层2
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            # 卷积层3
            nn.Conv2d(out_channel, out_channel*dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel*dim)
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel*dim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel*dim)
            )
        self.relu = nn.ReLU()

    def forward(self, x):

        x_shortCut = x
        out = self.bottleneck(x)

        if self.downsampling:
            x_shortCut = self.downsample(x)

        out += x_shortCut
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, blocks, num_classes=2):
        super(ResNet50, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.layer2 = self.make_layer(64, 64, blocks[0], stride=1)
        self.layer3 = self.make_layer(256, 128, blocks[1], stride=2)
        self.layer4 = self.make_layer(512, 256, blocks[1], stride=2)
        self.layer5 = self.make_layer(1024, 512, blocks[1], stride=2)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, in_channel, out_channel, block_num, stride=1):
        layers = list()
        layers.append(Bottleneck(in_channel, out_channel, stride, downsampling=True))
        for i in range(1, block_num):
            layers.append(Bottleneck(out_channel*4, out_channel))
        # return Bottleneck(in_channel, out_channels, stride, downsampling=True)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    net = ResNet50([3, 4, 6, 3])
    summary(net, (3, 224, 224))
