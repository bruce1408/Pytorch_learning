import torch
from torch.autograd import Function

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = torch.tensor([4., 5., 6.], requires_grad=True)

z = torch.pow(x, 2) + torch.pow(y, 2)
f = z + x + y
s = 6 * f.sum()

print(f)
print(s)
s.backward()
print(x)
print(x.grad)