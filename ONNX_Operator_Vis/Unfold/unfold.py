import torch.onnx
import torch
import torch.nn as nn

# 创建一个包含unfold操作的PyTorch模型
class UnfoldModel(nn.Module):
    def __init__(self):
        super(UnfoldModel, self).__init__()
        self.unfold = nn.Unfold((2, 2), stride=2)

    def forward(self, x):
        return self.unfold(x)

# 创建模型实例
model = UnfoldModel()

# 创建示例输入张量
input_data = torch.randn(1, 1, 5, 5)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
# onnx_path = "unfold_model.onnx"
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/unfold.onnx"

torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
