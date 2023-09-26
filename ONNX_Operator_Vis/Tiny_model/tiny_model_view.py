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
            x1 = self.conv1(inputs)
            x2 = self.conv2(x1)
            # 带动态输入的 view 或者 reshape 转成 onnx 会有shape/gather/unsqueeze/concat算子。
            x2_flatten = x2.view(x2.size(0), -1)
            x2_flatten = torch.reshape(x2, (x2.size(0), -1))
            print(x2_flatten.shape)
            # x2_flatten = torch.flatten(x2, start_dim=1)
            x3 = self.fc1(x2_flatten)
            x4 = self.fc2(x3)        
            return x4



# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(1, 3, 128, 128)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/tiny_model.onnx"
torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
