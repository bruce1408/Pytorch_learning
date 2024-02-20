import torch.onnx
import torch
import torch.nn as nn

# gather 就是按照索引去拿值
class ClipOperator(nn.Module):
    def __init__(self):
        super(ClipOperator, self).__init__()
        
    def forward(self, a1):
        output = torch.clamp(a1, min=-1, max=1)
        # output = self.clip(a1)
        print(output.shape)
        return output



# 创建模型实例
model = ClipOperator()

# 创建示例输入张量
input_data = torch.randn(1, 3, 224, 224)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "./clip.onnx"
torch.onnx.export(model, (input_data), onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)
