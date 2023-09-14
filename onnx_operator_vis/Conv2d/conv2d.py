import os
import torch
import torch.nn as nn
import torch.onnx as onnx


# 可有可无，不是重点。定义一个简单的PyTorch模型，可以换成你自己的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv_layer = nn.Conv2d(3, 16, 3, stride=2)
    def forward(self, x):
        print(x.shape)
        x  = self.conv_layer(x)
        print(x.shape)
        
        return x


# 可有可无，不是重点。创建一个示例模型实例，如果你有pth文件可以在这里加载
model = SimpleModel()

# 定义输入张量，这个要关注一下，张量的形状必须符合你模型的要输入的模型的张量的形状，这个input会在模型里完整的跑一遍
input_tensor = torch.randn(1, 3, 224, 224)

if __name__=="__main__":
    # 导出模型为ONNX格式
    operator_name = "conv2d"
    model_dir = "/root/bruce_cui/onnx_operator_vis/onnx_operators"
    onnx_file_path = "conv2d.onnx"
    onnx.export(model, input_tensor, os.path.join(model_dir, onnx_file_path), opset_version=11, verbose=True)

