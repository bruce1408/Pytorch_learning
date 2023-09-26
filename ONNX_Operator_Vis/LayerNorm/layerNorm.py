import torch.onnx
import torch
import torch.nn as nn

batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
        

# 创建一个包含ConvTranspose2d操作的PyTorch模型
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        self.avgpool = nn.BatchNorm2d(100)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        output = self.layer_norm(x)
        print(output.shape)
        return output
    
# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
N, C, H, W = 20, 5, 10, 10
input_data = torch.randn(N, C, H, W)
# input_data = torch.randn(20, 100, 35, 45)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/layernorm.onnx"
torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
