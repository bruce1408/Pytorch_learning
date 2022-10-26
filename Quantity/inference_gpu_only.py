import os
import torch
import time
import numpy as np
# from torchvision.models import resnet18
from torchvision.models import MobileNetV2
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet
from torchvision.models import vgg11_bn

if __name__ == '__main__':
    # model = EfficientNet.from_pretrained('efficientnet-b3')
    # model = resnet18(pretrained=False)
    # model = resnet50(pretrained=False)
    model = vgg11_bn(pretrained=False)
    # model = MobileNetV2()
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    dump_input = torch.ones(32, 3, 224, 224).to(device)

    # Warn-up
    for _ in range(5):
        start = time.time()
        outputs = model(dump_input)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end - start) * 1000))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                         profile_memory=False) as prof:
        outputs = model(dump_input)
    print(prof.table())

    # 打开浏览器，然后输入 chrome://tracing/
    prof.export_chrome_trace('./resnet_profile.json')
