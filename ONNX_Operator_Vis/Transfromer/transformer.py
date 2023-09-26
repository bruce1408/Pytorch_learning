import torch.onnx
import torch
import torch.nn as nn

# 创建一个包含ConvTranspose2d操作的PyTorch模型
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        self.avgpool = nn.Transformer(nhead=16, num_encoder_layers=12)
        

    def forward(self, x, y):
        return self.avgpool(x, y)

# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
# input_data = torch.randn(20, 16, 28, 28)  # 输入形状为(1, 1, 5, 5)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
# >>> out = transformer_model(src, tgt)

# 导出模型到ONNX格式
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/transformer.onnx"
torch.onnx.export(model, (src, tgt), onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
