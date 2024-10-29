import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# 1. 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224x224
    transforms.ToTensor(),          # 将图像转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
])

# 2. 创建随机输入张量
input_tensor = torch.randn(1, 3, 224, 224)  # 创建一个随机输入张量

# 3. 加载预训练的 ResNet-34 模型
# 这里我们使用 torchvision 提供的 ResNet-34 模型作为替代
model = models.resnet34(pretrained=True)

# 修改最后的全连接层以适应类别数量
num_classes = 1000  # 设置你的类别数量
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.eval()  # 设置为评估模式

# 4. 导出为 ONNX 模型
onnx_path = "/mnt/share_disk/bruce_trie/misc_data_products/onnx_models/resnet34_model.onnx"

dummy_input = torch.randn(1, 3, 224, 224)  # 创建一个假输入，用于导出模型

torch.onnx.export(
    model,                      # 要导出的模型
    dummy_input,                # 模拟输入
    onnx_path,                  # 输出 ONNX 模型的路径
    input_names=['input'],      # 输入张量的名称
    output_names=['output'],    # 输出张量的名称
    opset_version=11            # ONNX opset 版本
)

print(f"ONNX 模型已成功导出到: {onnx_path}")
