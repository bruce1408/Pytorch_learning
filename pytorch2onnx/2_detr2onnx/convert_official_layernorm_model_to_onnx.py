import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from toymodel_with_layernorm import CustomModel


model = CustomModel()
x = torch.randn(1, 3, 224, 224)

state_dict = torch.load("models/official_layernorm.pth")['model']
model.load_state_dict(state_dict)
model.eval()


with torch.no_grad():
    torch.onnx.export(model, 
                      x, 
                      "official_layernorm.onnx", 
                      opset_version=12,
                      input_names=["input"],
                      output_names=["output"]
                      )




# export onnx model
# export_name = './trained_official_layernorm.onnx'
# torch.onnx.export(CustomModel(), x, 
#                     export_name,
#                     opset_version=11,
#                     input_names=["X"], 
#                     output_names=["Y"]
#                     )