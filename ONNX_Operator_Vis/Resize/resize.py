import torch.onnx
import torch
import torch.nn as nn

# resize 其实就是插值运算按照索引去拿值
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        
    def forward(self, a1):
        output = torch.nn.functional.interpolate(a1, scale_factor=2.0, mode='bilinear')
        print(output.shape)
        return output



# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(1, 3, 128, 128)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/resize_interpolate.onnx"
torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode='bilinear')

    def forward(self, x):
        return self.upsample(x)

model = MyModel()
input_data = torch.randn(1, 3, 224, 224)

# onnx_path = "upsample_model.onnx"
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/resize_unsample.onnx"

torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)
