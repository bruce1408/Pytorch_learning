import torch
import torch.nn as nn
from torch.autograd import Function
"""
https://blog.csdn.net/t20134297/article/details/107870906
https://blog.csdn.net/yskyskyer123/article/details/94905856
https://github.com/fungtion/DANN
"""

# 梯度不反转
# x = torch.tensor([1., 2., 3.], requires_grad=True)
# y = torch.tensor([4., 5., 6.], requires_grad=True)
#
# z = torch.pow(x, 2) + torch.pow(y, 2)
# f = z + x + y
# s = 6 * f.sum()
# print(z)
# print(f)
# print(s)
# s.backward()
# print(x)
# print(x.grad)


x = torch.tensor([1., 2., 3.], requires_grad=True)
y = torch.tensor([4., 5., 6.], requires_grad=True)
z = torch.pow(x, 2) + torch.pow(y, 2)

f = z + x + y


class GRL(Function):
    @staticmethod
    def forward(self, input):
        return input

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.neg()
        return grad_input


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# 第一种方法
Grl = GRL()
s = 6 * f.sum()
s = Grl.apply(s)
print('s: ', s)
s.backward()
print('x: ', x)
print('x grad: ', x.grad)


# 第二种方式
# discritor = nn.Sequential(
#     GradientReversal()
# )
# s = 6 * f.sum()
# s = discritor(s)
# s.backward()
# print(x)
# print(x.grad)



