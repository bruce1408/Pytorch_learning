import torch
import torch.nn as nn

m = nn.AdaptiveAvgPool2d((52, 52))
input = torch.randn(1, 3, 224, 224)
output = m(input)
print(output.shape)