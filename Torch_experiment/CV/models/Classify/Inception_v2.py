import torch
import torch.nn as nn
from torchsummary import summary


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class inception_block(nn.Module):
    """
    残差模块
    """
    def __init__(self, in_channel, out_channel):
        super(inception_block, self).__init__()
        # 1*1 conv
        self.conv1 = nn.Conv2d(in_channel, out_channel[0], 1, 1)
        # 1*1 3*3 conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel[1], 1, 1),
            nn.Conv2d(out_channel[1], out_channel[2], 3, 1, 1)
        )
        # 1*1, 3*3 conv
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel[3], 1, 1, 1),
            nn.Conv2d(out_channel[3], out_channel[4], 5, 1, 1)
        )
        # maxpoolConv
        self.maxpoolConv = nn.Sequential(
            nn.MaxPool2d(3, 1),
            nn.Conv2d(in_channel, out_channel[5], 1, 1, 1)
        )
        self.apply(weights_init)

    def forward(self, input):
        out1 = self.conv1(input)
        out3 = self.conv3(input)
        out5 = self.conv5(input)
        pool = self.maxpoolConv(input)
        out = torch.cat([out1, out3, out5, pool], dim=1)
        return out


class aux_logits(nn.Module):
    """
    辅助损失函数模块
    """
    def __init__(self, in_channel, out_channel):
        super(aux_logits, self).__init__()
        self.conv2d = nn.Sequential(
            nn.AvgPool2d(5, 3),
            nn.Conv2d(in_channel, 128, 1, 1)

        )
        self.aux_logits = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, out_channel),
            nn.Softmax(dim=1)
        )

    def forward(self, input):

        x = self.conv2d(input)
        x = x.view(x.shape[0], -1)
        x = self.aux_logits(x)
        return x


class Inception_v1(nn.Module):
    """
    inception v1 代码块
    """
    def __init__(self, num_classes=10):
        super(Inception_v1, self).__init__()
        self.mode = 'train'
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(64, 192, 1, 1),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.MaxPool2d(3, 2, padding=1)
        )  # 192

        self.layer2 = nn.Sequential(
            inception_block(192, [64, 96, 128, 16, 32, 32]),
            inception_block(256, [128, 128, 192, 32, 96, 64]),  # 480
            nn.MaxPool2d(3, 2, 1)
        )

        # aux loss1
        self.layer3_1 = nn.Sequential(
            inception_block(480, [192, 96, 208, 16, 48, 64])     # 512
        )

        self.aux_logits1 = aux_logits(512, num_classes)

        self.layer3_2 = nn.Sequential(
            inception_block(512, [160, 112, 224, 24, 64, 64]),
            inception_block(512, [128, 128, 256, 24, 64, 64]),
            inception_block(512, [112, 144, 288, 32, 64, 64]),
        )

        self.aux_logits2 = aux_logits(528, num_classes)

        self.layer3_3 = nn.Sequential(
            inception_block(528, [256, 160, 320, 32, 128, 128]),  # 832
            nn.MaxPool2d(3, 2, 1),
            inception_block(832, [256, 160, 320, 32, 128, 128]),
            inception_block(832, [384, 192, 384, 48, 128, 128]),
            nn.AvgPool2d(7, 1),
            nn.Dropout(0.2),
        )

        self.linear = nn.Linear(1024, num_classes)
        self.sofmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        aux1 = x = self.layer3_1(x)
        aux2 = x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = x.view(x.shape[0], -1)
        out = self.sofmax(self.linear(x))
        aux1 = self.aux_logits1(aux1)
        aux2 = self.aux_logits2(aux2)
        return aux1, aux2, out


if __name__ == '__main__':

    net = Inception_v1(2)
    if torch.cuda.is_available():
        summary(net.cuda(), (3, 224, 224))
    else:
        summary(net, (3, 224, 224))