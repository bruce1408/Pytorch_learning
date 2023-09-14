import torch.onnx as onnx
import os
import torch
import torch.nn as nn

# 创建一个包含ConvTranspose1d操作的PyTorch模型
class ConvTranspose1dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose1dModel, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3)

    def forward(self, x):
        output = self.conv_transpose(x)
        print(output.shape)
        return output

# 创建模型实例
model = ConvTranspose1dModel()

# 创建示例输入张量
input_data = torch.randn(1, 1, 5)  # 输入形状为(1, 1, 5)


# 定义输入张量，这个要关注一下，张量的形状必须符合你模型的要输入的模型的张量的形状，这个input会在模型里完整的跑一遍
input_tensor = torch.randn(20, 16, 50)

if __name__=="__main__":
    # 导出模型为ONNX格式
    model_dir = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators"
    onnx_file_path = "convtranspose1d.onnx"
    onnx.export(model, input_data, os.path.join(model_dir, onnx_file_path), opset_version=11, verbose=True)

