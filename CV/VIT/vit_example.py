import os
import torch
from vit_pytorch import SimpleViT

v = SimpleViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=12,
    mlp_dim=2048
)

img = torch.randn(4, 3, 224, 224)

preds = v(img)  # (1, 1000)
print(preds.shape)

print(torch.__version__)
