import os
import torch
import torch.nn as nn
import torch.onnx as onnx


# 可有可无，不是重点。定义一个简单的PyTorch模型，可以换成你自己的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.unfold = nn.Unfold((2,2), stride=2)
        
    def forward(self, x):
        print(x.shape)
        x  = self.unfold(x)
        print(x.shape)
        
        return x



x = torch.Tensor([[[[  1,  2,  3,  4],
   					[  5,  6,  7,  8],
   					[  9, 10, 11, 12],
   					[ 13, 14, 15, 16]]]])


# 创建模型实例
model = SimpleModel()

# 导出模型到ONNX格式
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/convtranspose2d.onnx"
torch.onnx.export(model, x, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)

