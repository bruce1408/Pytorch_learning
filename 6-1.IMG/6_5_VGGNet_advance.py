"""
VGG in pytorch
VGG-11/13/16/19 in Pytorch.
Very Deep Convolutional Networks for Large-Scale Image Recognition.
https://arxiv.org/abs/1409.1556v6
"""
import torch
import torch.nn as nn
from torchsummary import summary

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        print(output.shape)
        output = output.view(output.shape[0], -1)
        output = self.classifier(output)
        return output


def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for layerType in cfg:
        if layerType == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, layerType, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(layerType)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = layerType

    return nn.Sequential(*layers)


def vgg11_bn():
    return VGG(make_layers(cfg['vgg11'], batch_norm=True))


def vgg13_bn():
    return VGG(make_layers(cfg['vgg13'], batch_norm=True))


def vgg16_bn():
    return VGG(make_layers(cfg['vgg16'], batch_norm=True))


def vgg19_bn():
    return VGG(make_layers(cfg['vgg19'], batch_norm=True))


if __name__ == '__main__':
    net = vgg16_bn()
    print(net)
    if torch.cuda.is_available():
        summary(net.cuda(), (3, 224, 224))
    else:
        summary(net, (3, 224, 224))
