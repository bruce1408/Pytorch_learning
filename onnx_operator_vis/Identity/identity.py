import torch.onnx
import torch
import torch.nn as nn

# identity 不做任何操作，输入是什么，输出就是什么，一般用来是进行残差网络的
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        self.identity = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)

    def forward(self, x):
        return self.identity(x)

# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(20, 16, 28, 28)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/identity.onnx"
torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
