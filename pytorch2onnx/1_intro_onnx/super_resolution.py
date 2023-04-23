import os
import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

    # Download checkpoint and test image


# urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
#         'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png']
# names = ['srcnn.pth', 'face.png']
# for url, name in zip(urls, names):
#     if not os.path.exists(name):
#         open(name, 'wb').write(requests.get(url).content)


def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)

    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


model = init_torch_model()
input_img = cv2.imread('../data/face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img)).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("face_torch.png", torch_output)

x = torch.randn(1, 3, 256, 256)

# 第一个参数是要转换的模型、接下来是模型的任意一组输入、到处onnx文件的文件名
# with torch.no_grad():
#     torch.onnx.export(
#         model,
#         x,
#         "srcnn.onnx",
#         opset_version=11, # 表示onnx算子集
#         input_names=['input'],  # 输入
#         output_names=['output'])  # 模型输出


# 添加runtime运行时代码
import onnxruntime

# 获取一个onnx runtime 推理器，用于进行onnx模型推理
ort_session = onnxruntime.InferenceSession("srcnn.onnx")
ort_inputs = {'input': input_img}

# run方法适用于模型推理，第一个参数为输出张量名的列表；第二个采纳数是输入值的字典
# 其中输入值字典的 key 为张量名，value 为 numpy 类型的张量值。
# 输入输出张量的名称需要和torch.onnx.export 中设置的输入输出名对应
ort_output = ort_session.run(['output'], ort_inputs)[0]

# 如果代码正常运行的话，另一幅超分辨率照片会保存在"face_ort.png"中。
# 这幅图片和刚刚得到的"face_torch.png"是一模一样的。
# 这说明 ONNX Runtime 成功运行了 SRCNN 模型，模型部署完成了
ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_ort.png", ort_output)