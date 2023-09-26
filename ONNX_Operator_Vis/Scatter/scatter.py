import torch.onnx
import torch
import torch.nn as nn

# 创建一个示例目标张量
data = torch.tensor([1.0, 2.0, 3.0, 4.0])

# gather 就是按照索引去拿值
class ConvTranspose2dModel(nn.Module):
    def __init__(self):
        super(ConvTranspose2dModel, self).__init__()
    
    
    def forward(self, data):
        
        # 创建一个示例索引张量
        indices = torch.tensor([0, 2, 1, 3])

        # 创建一个示例值张量
        values = torch.tensor([10.0, 20.0, 30.0, 40.0])

        # 使用 torch.scatter_ 将值散布到目标张量的指定位置
        data.scatter_(0, indices, values)
        # print(data.shape)
        # print(data)

        return data

# 创建模型实例
model = ConvTranspose2dModel()

# 导出模型到ONNX格式
onnx_path = "/root/bruce_cui/onnx_operator_vis/ONNX_Operators/scatter.onnx"
torch.onnx.export(model, data, onnx_path, verbose=True, opset_version=11)
