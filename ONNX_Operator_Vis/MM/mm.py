import torch.onnx as onnx
import torch
import torch.nn as nn
import os

# 创建一个包含torch.mm,矩阵乘法
class MatMulModel(nn.Module):
    def __init__(self):
        super(MatMulModel, self).__init__()

    def forward(self, x, y):
        output = torch.mm(x, y)
        print(output.shape)
        return output
# 创建模型实例
model = MatMulModel()

# 创建示例输入张量
input_data1 = torch.randn(3, 4)  # 输入1形状为(1, 3, 4)
input_data2 = torch.randn(4, 2)  # 输入2形状为(1, 4, 2)

# 导出模型到ONNX格式


if __name__=="__main__":
    # 导出模型为ONNX格式
    model_dir = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators"
    onnx_file_path = "mm.onnx"
    onnx.export(model, (input_data1, input_data2), os.path.join(model_dir, onnx_file_path), opset_version=11, verbose=True)

