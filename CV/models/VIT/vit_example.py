import os
import torch
from vit_pytorch import ViT
from torchsummary import summary
from thop import profile

# 对应的就是model的large
v = ViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=1024,
    depth=24,
    heads=16,
    mlp_dim=4096
)


# v = ViT(
#     image_size=224,
#     patch_size=16,
#     num_classes=1000,
#     dim=1280,
#     depth=32,
#     heads=16,
#     mlp_dim=5120
# )

img = torch.randn(4, 3, 224, 224)

# preds = v(img)  # (1, 1000)
# print(preds.shape)


if torch.cuda.is_available():
    summary(v.cuda(), (3, 224, 224))
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(v.cuda(), (dummy_input,))
    print('FLOPs: %.3f G, params: %.2f M' % (flops / 1e9, params / 1000000.0))

else:
    summary(v, (3, 224, 224))
