import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from toymodel_with_layernorm import CustomModel
from torch.onnx import register_custom_op_symbolic
from typing import Union, List, Tuple
from torch import Tensor, Size


def register_custom_op():
    def my_layer_norm(g, input, weight, bias, eps):
        return g.op("customdomain::testlayernorm", input,  weight, bias, epsilon_f=0.0001)
    register_custom_op_symbolic("myname::custom_layer_norm", my_layer_norm, 9)

_shape_t = Union[int, List[int], Size]

class CustomOpNormLayerVersion(torch.nn.Module):
    
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomOpNormLayerVersion, self).__init__()
    
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        return torch.ops.myname.custom_layer_norm(x, self.weight, self.bias, 0.001)


# 定义模型
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 1, bias=False)  
        self.bn = nn.BatchNorm2d(256)
        # self.layernorm = nn.LayerNorm(256)
        self.layernorm = CustomOpNormLayerVersion([256])

        self.fc = nn.Linear(25690112, 3)

    def forward(self, x):
        batch_size = x.shape[0]
        output = self.conv1(x)
        output = self.bn(output)
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(1, -1, 256)
        output = self.layernorm(output)
        output = F.interpolate(output, scale_factor=2, mode='nearest')
        output = output.reshape(batch_size, -1)
        output = self.fc(output)
        return output

torch.ops.load_library(
"/home/cuidongdong/onnxruntime_deploy/custom_operator/custom_operator_layernorm/build/lib.linux-x86_64-cpython-39/custom_layer_norm.cpython-39-x86_64-linux-gnu.so")
register_custom_op()
model = CustomModel()
x = torch.randn(1, 3, 224, 224)

state_dict = torch.load("/home/cuidongdong/onnxruntime_deploy/models/official_layernorm.pth")['model']
model.load_state_dict(state_dict)
model.eval()


with torch.no_grad():
    torch.onnx.export(model, 
                      x, 
                      "custom_layernorm.onnx", 
                      opset_version=12,
                      input_names=["input"],
                      output_names=["output"]
                      )
