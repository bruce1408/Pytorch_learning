import os
import sys
import onnx
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
from onnxsim import simplify


def onnx_simplify(path):
    # simplify the onnx model & load onnx model
    onnx_model = onnx.load(path)  
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    
    output_path = path.split(".")[0] + "_onnxsim." + path.split(".")[-1]
    
    onnx.save(model_simp, output_path)
    
    print("The simplified ONNX model is saved to {}".format(output_path))


class Net(nn.Module):
    """
    实现一个简单的只有三层卷积的神经网络来做训练.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 7, 2, 3),
            # nn.AdaptiveAvgPool2d((112, 112)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 1, 2),
            nn.Conv2d(256, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, 2, 1)  # 7
            
        )
        # self.exp = torch.exp()
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self.input_1 = torch.Tensor(1, 2)
        self.input_2 = torch.Tensor(1, 2)

    def forward(self, input1):
        output = torch.add(self.input_1, self.input_2)
        return output
        # output = self.net(input1)
        # print("output shappe is: ", output.shape)
        # output = output.reshape(output.shape[0], -1)
        # output = self.fc1(output)
        # output_drop = self.drop(output)
        # output_sum  = torch.add(output , output_drop)
        # output_sum[0:, ] += output[0:,]
        # print("before mm is: ", output_sum.shape, output.transpose(0, 1).shape)
        # output_mm = torch.mm(output_sum, output.transpose(0, 1))
        # temp_add = torch.add(output_sum, 0.0001)
        # print(output_mm.shape, output_drop.shape)
        # temp_mm = torch.mm(output_mm, output_drop)
        # output_mul = torch.div(temp_mm, temp_add)
        # output = self.fc2(output_mul)
        # output_drop_2 = self.drop(output)
        # output_sub = torch.sub(output, output_drop_2)
        # output = self.fc3(output_sub)
        # output = torch.exp(output)
        # output = output[0]
        # return temp_mm


if __name__ == '__main__':
    model = Net()
   

    # ============ model structure ===========
    # if torch.cuda.is_available():
    #     summary(model.cuda(), (3, 224, 224))
    # else:
    #     summary(model, (3, 224, 224))
    
    # 创建示例输入张量
    input_data = torch.randn(1, 3, 224, 224)  

    # 导出模型到ONNX格式
    onnx_path = "/Users/bruce/PycharmProjects/Pytorch_learning/ONNX_Operator_Vis/ONNX_Operators/tiny_model_conv3_s1_maxpool2_batch_tensor.onnx"
    torch.onnx.export(model, input_data, onnx_path, verbose=True, opset_version=11)
    
    # 使用onnxsim去消掉unsqueeze
    onnx_simplify(onnx_path)
