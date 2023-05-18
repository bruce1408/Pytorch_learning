import os
import numpy as np
import torch
from torchvision.models import resnet18
import time


if __name__ == '__main__':
    model = resnet18(pretrained=False)
    device = torch.device('cpu')
    model.eval()
    model.to(device)
    dump_input = torch.ones(1, 3, 224, 224).to(device)

    # Warn-up
    # torch.cuda.synchronize()
    for _ in range(5):
        start = time.time()
        outputs = model(dump_input)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))

    with torch.autograd.profiler.profile(enabled=True,
                                         use_cuda=False,
                                         record_shapes=False,
                                         profile_memory=False) as prof:
        outputs = model(dump_input)
    print(prof.table())

