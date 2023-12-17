import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class Dense_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


# 0.4.0版本,兼容1.3.0版本
# class GradReverse(torch.autograd.Function):
#     def __init__(self, lambd):
#         self.lambd = lambd
#
#     def forward(self, x):
#         return x.view_as(x)
#
#     def backward(self, grad_output):
#         grad_ = grad_output.clone()
#         return grad_ * - self.lambd

# 修改为1.5.0版本
# class GradReverse(torch.autograd.Function):
#     # @staticmethod
#     # def __init__(self, lambd):
#     #     super(GradReverse, self).__init__()
#     #     self.lambd = lambd
#     @staticmethod
#     def forward(ctx, x):
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         # grad_ = grad_output.clone()
#         return grad_output.neg()
#
#
# def grad_reverse(x, lambd=1.0):
#     return GradReverse.apply(x)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input

# class GradReverse(Function):
#     @staticmethod
#     def forward(ctx, x, lambd):
#         ctx.save_for_backward(lambd)
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         lambd=ctx.saved_tensors[0]
#         return grad_output.neg()*lambd, None

# class GradReverse(Function):
#
#     # 重写父类方法的时候，最好添加默认参数，不然会有warning
#     @ staticmethod
#     def forward(ctx, x, lambd, **kwargs: None):
#         ctx.lambd = lambd
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, *grad_output):
#         return grad_output * -ctx.lambd, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x)


class Generator_s2m(nn.Module):
    def __init__(self):
        super(Generator_s2m, self).__init__()
        self.conv1 = Conv_Block(3, 64, kernel_size=5)
        self.conv2 = Conv_Block(64, 64, kernel_size=5)
        self.conv3 = Conv_Block(64, 128, kernel_size=3, stride=2)
        self.conv4 = Conv_Block(128, 128, kernel_size=3, stride=2)
        self.fc1 = Dense_Block(3200, 100)
        self.fc2 = Dense_Block(100, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Classifier_s2m(nn.Module):
    def __init__(self, n_output):
        super(Classifier_s2m, self).__init__()
        self.fc = nn.Linear(100, n_output)

    def forward(self, x):
        x = self.fc(x)
        return x


class Generator_u2m(nn.Module):
    def __init__(self):
        super(Generator_u2m, self).__init__()
        self.conv1 = Conv_Block(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = Conv_Block(20, 50, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.drop = nn.Dropout()
        self.fc = Dense_Block(800, 500)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


class Classifier_u2m(nn.Module):
    def __init__(self, n_output):
        super(Classifier_u2m, self).__init__()
        self.fc = nn.Linear(500, n_output)

    def forward(self, x):
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, task='s2m'):
        super(Net, self).__init__()
        if task == 's2m':
            self.generator = Generator_s2m()
            self.classifier = Classifier_s2m(6)
        elif task == 'u2m' or task == 'm2u':
            self.generator = Generator_u2m()
            self.classifier = Classifier_u2m(6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant=1, adaption=False):
        x = self.generator(x)
        if adaption == True:
            x = grad_reverse(x, constant)
        x = self.classifier(x)
        return x
