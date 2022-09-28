from torchvision.models import resnet50
from thop import profile
import torch
model = resnet50(pretrained="/Volumes/Elements/PycharmProjects/3D_model/bevdet/resnet50-0676ba61.pth")
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
print(macs)
print(params)