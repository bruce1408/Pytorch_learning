# coding=UTF-8
import torch
import math
import torchvision.models as models
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision.models import vgg11, resnet50

"""
加载部分的预训练模型,首先写一个自己的模型,保留想要迁移过来的网络结果名称,然后在加上自己定制的网络结构层,这里以resnet50为例子
resnet50官网代码地址:https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18
"""


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
        直接使用原来torch官网的例子的功能代码模块
        :param inplanes:
        :param planes:
        :param stride:
        :param downsample:
        :param groups:
        :param base_width:
        :param dilation:
        :param norm_layer:
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CNN(nn.Module):

    def __init__(self, block, layers, num_classes=9):
        """
        自己定义的几层进行训练,保留想要使用预训练部分的层数的权重,就要名字起的和预训练网络结构的名字一样,比如self.conv2, self.bn1,
        self.layer1等,加上自己想要定制和训练的特殊层.最后模型会迁移预训练网络结构里面的名字一样的层的参数.然后只训练自己的层数.
        :param block:
        :param layers:
        :param num_classes:
        """
        self.inplanes = 64
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 新增一个反卷积层
        self.convtranspose1 = nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=1, output_padding=0,
                                                 groups=1, bias=False, dilation=1)
        # 新增一个最大池化层
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 去掉原来的fc层，新增一个fclass层
        self.fclass = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # 新加层的forward
        x = x.view(x.size(0), -1)
        x = self.convtranspose1(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fclass(x)

        return x


# 加载model
resnet50 = models.resnet50(pretrained=True)
cnn = CNN(Bottleneck, [3, 4, 6, 3])
print("="*10 + "before convert the params the value is: \n", cnn)
print("="*10 + ' the model resnet50 is: \n', resnet50)

# 读取参数
pretrained_dict = resnet50.state_dict()
model_dict = cnn.state_dict()

# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# 更新现有的model_dict
model_dict.update(pretrained_dict)

# 加载我们真正需要的state_dict, 替换了原来的 avgpool 层和 fc 层
cnn.load_state_dict(model_dict)
print("="*10+" after convert the params the value is: \n", cnn)
