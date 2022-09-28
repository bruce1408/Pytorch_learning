import os
import torch
from vit_pytorch import ViT
from torchsummary import summary

# 对应的就是model的large
# v = ViT(
#     image_size=224,
#     patch_size=16,
#     num_classes=1000,
#     dim=1024,
#     depth=24,
#     heads=16,
#     mlp_dim=4096
# )


v = ViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=1280,
    depth=32,
    heads=16,
    mlp_dim=5120
)

img = torch.randn(4, 3, 224, 224)

# preds = v(img)  # (1, 1000)
# print(preds.shape)

print(torch.__version__)

if torch.cuda.is_available():
    summary(v.cuda(), (3, 224, 224))
else:
    summary(v, (3, 224, 224))
