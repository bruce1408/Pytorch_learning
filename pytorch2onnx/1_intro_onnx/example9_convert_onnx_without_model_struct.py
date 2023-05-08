import onnx
import torch

# 加载PyTorch模型
model = torch.load('/Users/bruce/Downloads/Chip_test/Novauto_chip_test_results/npu_sample/sampleResnet50/QuantizeAndCompile/resnet50-19c8e357.pth')

# 准备示例输入张量
# input_shape = (1, 3, 224, 224)
# example_input = torch.randn(input_shape)

# # 将模型转换为ONNX格式
# with torch.no_grad():
#     torch.onnx.export(model, example_input, 'my_model.onnx', input_names=['input'], output_names=['output'])
