import torch
from torch.nn import functional as F
import torch.onnx
import torch.nn as nn

# 创建一个包含ConvTranspose2d操作的PyTorch模型
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
        self.avgpool = nn.AvgPool2d((3, 2), stride=(2, 1))
        self.padding =  (
            1,2,   # 前面填充1个单位，后面填充两个单位，输入的最后一个维度则增加1+2个单位，成为8
            2,3,
            3,4
        )

    def forward(self, x):
                
        b = F.pad(x, self.padding)
        print(b.shape)
        return b



# 创建模型实例
model = ConvTranspose2dModel()

# 创建示例输入张量
input_data = torch.randn(1, 3, 224, 224)  # 输入形状为(1, 1, 5, 5)

# 导出模型到ONNX格式
onnx_path = "/Users/bruce/PycharmProjects/Pytorch_learning/ONNX_Operator_Vis/ONNX_Operators/padding.onnx"
torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)

print("ONNX模型已导出到:", onnx_path)