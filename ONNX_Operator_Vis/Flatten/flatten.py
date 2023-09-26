import torch.onnx
import torch
import torch.nn as nn

# linear 
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        # self.identity = nn.Linear(20, 30)

    def forward(self, x):
        output = x.flatten()
        print(output.shape)
        return output

# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(1, 3, 223, 224)

# 导出模型到ONNX格式
onnx_path = "/Users/bruce/PycharmProjects/Pytorch_learning/ONNX_Operator_Vis/ONNX_Operators/flatten_4dim.onnx"
torch.onnx.export(model, input_data, onnx_path, verbose=False, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
