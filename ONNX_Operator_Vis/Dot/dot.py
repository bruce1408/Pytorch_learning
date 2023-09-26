import torch.onnx as onnx
import torch
import torch.nn as nn
import os

# 创建一个包含torch.dot, 一位点乘
class MatMulModel(nn.Module):
    def __init__(self):
        super(MatMulModel, self).__init__()

    def forward(self, x, y):
        output = torch.dot(x, y)
        print(output.shape)
        return output
# 创建模型实例
model = MatMulModel()

# 创建示例输入张量
a1 = torch.Tensor([1, 2, 3])
a2 = torch.Tensor([1, 2, 3])

# 导出模型到ONNX格式


if __name__=="__main__":
    # 导出模型为ONNX格式
    model_dir = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators"
    onnx_file_path = "dot.onnx"
    onnx.export(model, (a1, a2), os.path.join(model_dir, onnx_file_path), opset_version=11, verbose=True)

