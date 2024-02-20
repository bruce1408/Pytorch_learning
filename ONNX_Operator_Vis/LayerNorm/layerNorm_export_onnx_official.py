import torch.onnx
import torch
import torch.nn as nn

batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
        

# 创建一个包含LayerNormModel操作的PyTorch模型
class LayerNormModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.norm  = nn.LayerNorm(3)
        self.act   = nn.ReLU()

    def forward(self, x):
        _, _, H, W = x.shape
        L = H * W
        x = self.conv1(x)
        x = x.view(x.shape[0], x.shape[1], L).permute(0, 2, 1)
        x = self.norm(x)
        x = self.act(x)
        return x
    
# 创建模型实例
model = LayerNormModel()

# 创建示例输入张量
# N, C, H, W = 20, 5, 10, 10
# input_data = torch.randn(N, C, H, W)

input_data  = torch.Tensor(1, 3, 224, 224).uniform_(-1, 1)

# input_data = torch.randn(20, 100, 35, 45)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "./layernorm_with_official_opset_11.onnx"
torch.onnx.export(model, input_data, onnx_path, verbose=False, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
