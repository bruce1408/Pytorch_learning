import math
import torch
import torch.nn as nn
import math, sys, os
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchsummary import summary
from CV.utils.DataSet_train_val_test import CustomData
import torchvision.transforms as transforms
from torchvision.models import resnet50

"""
使用官方预训练好的神经网络来加载权重, 自定义一个神经网络B,然后把预训练好的权重赋值给B网络进行训练.
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
save_path = "./transform_resnet50.pt"
gamma = 0.96
num_workers = 4
batchsize = 32  # batch_size 不要太大
epochs = 100
learning_rate = 0.01


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding
    :param in_planes:
    :param out_planes:
    :param stride:
    :return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1470):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.layer5 = self._make_detnet_layer(in_channels=2048)  # output shape = [batch, 256, 14, 14]

        # self.avgpool = nn.AvgPool2d(14) # fit 448 input size
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_end = nn.Conv2d(256, 30, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(30)
        self.lastfc = nn.Linear(7 * 7 * 30, 2)

        for m in self.modules():  # 遍历模型
            if isinstance(m, nn.Conv2d):  # isinstance：m类型判断    若当前组件为 conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 正太分布初始化
            elif isinstance(m, nn.BatchNorm2d):  # 若为batchnorm
                m.weight.data.fill_(1)  # weight为1
                m.bias.data.zero_()  # bias为0

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 步长为2时，即第一次进入layer时，增加downsample层。
            # 或者inplans（输入通道数） 不等于 block.expansion倍的planes = (输出通道数）
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_detnet_layer(self, in_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256, block_type='B'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
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
        x = self.layer5(x)  # [batch_size, dim, h, w]
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        # x = F.sigmoid(x)  # 归一化到0-1
        x = torch.sigmoid(x)
        # x = x.view(-1,7,7,30)
        x = x.permute(0, 2, 3, 1)  # (-1,7,7,30)
        # print()
        x = x.contiguous().view(-1, 7 * 7 * 30)
        x = self.lastfc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def Resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


mean = [0.485, 0.456, 0.406]
std = [0.2459, 0.2424, 0.2603115]

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop((224, 224), padding=4),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


trainset = CustomData('/raid/bruce/datasets/dogs_cats/train', transform=transform_train)
valset = CustomData('/raid/bruce/datasets/dogs_cats/train', transform=transform_val,
                    train=False, val=True, test=False, splitnum=0.8)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=True, num_workers=num_workers)


def get_acc(pred, label):
    total = pred.shape[0]
    _, pred_label = pred.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total


def train(model, epoch):
    model.train()
    optimizer.zero_grad()
    scheduler.step()
    for batch_idx, (img, label) in enumerate(trainloader):
        image = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        acc = get_acc(out, label)
        sys.stdout.write('\033[1;36m \r>>Train Epoch:%d [%d|%d] loss:%f, lr:%f, acc:%f\033[0m' %
                         (epoch, batch_idx, len(trainloader), loss.mean(), scheduler.get_lr()[0], acc))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


def val(model, epoch):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(valloader):
            image = img.cuda()
            label = label.cuda()
            out = model(image)
            _, predicted = torch.max(out.data, 1)
            total += image.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
            sys.stdout.write('\033[1;35m \r>>Validation Epoch:%d [%d|%d] total:%d, corretc:%d \033[0m' %
                             (epoch, batch_idx, len(valloader), total, correct.numpy()))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


if __name__ == "__main__":
    # ================= 测试网络结构是否正确 =================
    net = Resnet50()
    if torch.cuda.is_available():
        summary(net.cuda(), (3, 224, 224))
        x = torch.rand((1, 3, 224, 224)).cuda()
        output = net(x).cuda()
    else:
        summary(net, (3, 224, 224))
    # =================== 记载预训练模型 ====================
    net = Resnet50()  # 自己定义的网络
    officialNet = resnet50(pretrained=True)  # 官方的网络结构

    # 对官方模型预训练参数进行固定
    for para in officialNet.parameters():
        para.requires_grad = False

    # 更新网络参数到自己定义的网络
    new_state_dict = officialNet.state_dict()
    dd = net.state_dict()
    for index, k in enumerate(new_state_dict.keys()):
        print('total param is: ', k)
        if k in dd.keys() and not k.startswith('fc'):
            print('include param is:', k, index)
            dd[k] = new_state_dict[k]
        else:
            print('='*10, k)
    net.load_state_dict(dd)
    # ==================== 梯度参数设置 ====================
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=0.001, momentum=0.9,
                                weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.96, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    # =================== 模型训练部分 =====================
    for epoch in range(epochs):
        train(net.cuda(), epoch)
        val(net.cuda(), epoch)
        torch.save(net.load_state_dict(), '6_11_6.pt')




