import torch.onnx
import torch
import torch.nn as nn

# 创建一个包含ConvTranspose2d操作的PyTorch模型
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3)

    def forward(self, x):
        return self.conv_transpose(x)

# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(1, 1, 5, 5)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/convtranspose2d.onnx"
torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
