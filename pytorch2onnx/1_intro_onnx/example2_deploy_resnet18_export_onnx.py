import os
import cv2 
import torch 
import io, onnx
from torch import nn 
from torch.nn.functional import interpolate 
import torch.onnx 
import numpy as np 
import torchvision.models as models
from onnxsim import simplify

# 加载模型网络结构和权重
torch_module = models.resnet18(pretrained=True)
buffer = io.BytesIO()

# 模型转onnx
torch.onnx.export(torch_module, torch.randn((1, 3, 224, 224)), buffer, opset_version=11)
onnx_model = onnx.load_from_string(buffer.getvalue())
buffer.close()

# model_simp, check = simplify(onnx_model)
# assert check, "Simplified ONNX model could not be validated"
output_path = "./resnet18_new_convert.onnx"
onnx.save(model_simp, output_path)
print("The simplified ONNX model is saved to {}".format(output_path))
