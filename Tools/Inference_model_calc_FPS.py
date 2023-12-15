import torch
import time
import random
import numpy as np
from efficientnet_pytorch import EfficientNet

def randomSeed(SEED):
   random.seed(SEED)
   np.random.seed(SEED)
   torch.manual_seed(SEED)
   torch.cuda.manual_seed(SEED)
   torch.backends.cudnn.deterministic = True


randomSeed(0)

model = EfficientNet.from_pretrained('efficientnet-b3')
device = torch.device('cuda')
model.to(device)
dummy_input = torch.randn(1, 3, 300, 300, dtype=torch.float).to(device)


def measure_inference_speed(model, data, max_iter=2000, log_interval=50):
   """
   计算模型推理时间的函数
   :param model:
   :param data:
   :param max_iter:
   :param log_interval:
   :return:
   """
   model.eval()

   # the first several iterations may be very slow so skip them
   num_warmup = 50
   pure_inf_time = 0
   fps = 0

   # benchmark with 2000 image and take the average
   for i in range(max_iter):

      torch.cuda.synchronize()
      start_time = time.perf_counter()

      with torch.no_grad():
         model(*data)

      torch.cuda.synchronize()
      elapsed = time.perf_counter() - start_time

      if i >= num_warmup:
         pure_inf_time += elapsed
         if (i + 1) % log_interval == 0:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
               f'Done image [{i + 1:<3}/ {max_iter}], '
               f'fps: {fps:.1f} img / s, '
               f'times per image: {1000 / fps:.1f} ms / img',
               flush=True)

      if (i + 1) == max_iter:
         fps = (i + 1 - num_warmup) / pure_inf_time
         print(f'Overall fps: {fps:.1f} img / s, '
               f'times per image: {1000 / fps:.1f} ms / img',
            flush=True)
         break
   return fps


measure_inference_speed(model, (dummy_input,))



