import torch
import numpy as np
import time
from efficientnet_pytorch import EfficientNet
#
optimal_batch_size = 430
model = EfficientNet.from_pretrained('efficientnet-b3')
device = torch.device('cuda')
model.to(device)
# torch.seed()
dummy_input = torch.ones(optimal_batch_size, 3, 300, 300, dtype=torch.float).to(device)
repetitions = 100
total_time = 0
with torch.no_grad():
    for rep in range(repetitions):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) / 1000
        total_time += curr_time
Throughput = (repetitions * optimal_batch_size) / total_time
print('Final Throughput:', Throughput)


