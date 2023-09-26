import torch.onnx
import torch
import torch.nn as nn

# linear 
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        self.fc1 = nn.Linear(246016, 128)
        self.fc2 = nn.Linear(128, 64)
        self.conv1 = nn.Conv2d(3, 16, 3, 2)
        self.conv2 = nn.Conv2d(16, 256, 3, 2)

    def forward(self, inputs):
            
            B, C, H, W = inputs.shape
            return B*C*H*W



# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(1, 3, 128, 128)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/shape.onnx"
torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
