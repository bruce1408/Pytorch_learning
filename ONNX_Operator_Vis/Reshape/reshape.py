import torch.onnx
import torch
import torch.nn as nn

# linear 
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
    
    def forward(self, inputs):
            
            output = inputs.reshape(1, -1)
            return output



# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(1, 3, 128, 128)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "/Users/bruce/PycharmProjects/Pytorch_learning/onnx_operator_vis/ONNX_Operators/reshape.onnx"
torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
