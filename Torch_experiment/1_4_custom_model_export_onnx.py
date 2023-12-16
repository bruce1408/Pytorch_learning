# import torch
# import onnx
# import onnx.utils
# import onnx.optimizer

# class Model(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Model, self).__init__()
#         self.fc1 = torch.nn.Linear(input_size, hidden_size)
#         self.fc2 = torch.nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         return x

# # 创建模型实例
# input_size = 2
# hidden_size = 4
# output_size = 1
# model = Model(input_size, hidden_size, output_size)

# # 创建输入张量
# x = torch.randn(3, input_size)

# # 计算模型输出
# y = model(x)

# # 打印模型输出
# print(y)

# # 导出ONNX模型
# torch.onnx.export(model, x, "model.onnx", input_names=["input"], output_names=["output"], keep_initializers_as_inputs=True)

# # 加载导出的ONNX模型
# onnx_model = onnx.load("model.onnx")

# # 将Linear层转换为MatMul算子
# # add_value_info_for_constants(onnx_model)
# onnx_model = onnx.optimizer.optimize(onnx_model, passes=["fuse_matmul_add_bias_into_gemm"])

# # 保存转换后的ONNX模型
# onnx.save(onnx_model, "model_with_matmul.onnx")


import torch
import onnx
import onnx.utils
import onnx.optimizer

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# 创建模型实例
input_size = 2
hidden_size = 4
output_size = 1
model = Model(input_size, hidden_size, output_size)

# 创建输入张量
x = torch.randn(3, input_size)

# 计算模型输出
y = model(x)

# 打印模型输出
print(y)

# 导出ONNX模型
torch.onnx.export(model, x, "model.onnx", input_names=["input"], output_names=["output"], keep_initializers_as_inputs=True)

# 加载导出的ONNX模型
onnx_model = onnx.load("model.onnx")

# 将Linear层手动转换为MatMul算子
for node in onnx_model.graph.node:
    if node.op_type == "Gemm":
        weight_name = node.input[1]
        weight = None
        for init_node in onnx_model.graph.initializer:
            if init_node.name == weight_name:
                weight = init_node
                break
        if weight is not None:
            matmul_node = onnx.helper.make_node("MatMul", [node.input[0], weight.name], [node.output[0]])
            onnx_model.graph.node.remove(node)
            onnx_model.graph.node.append(matmul_node)

# 优化ONNX模型
onnx_model = onnx.optimizer.optimize(onnx_model, ["fuse_matmul_add_bias_into_gemm"])

# 保存ONNX模型
onnx.save(onnx_model, "model_with_matmul.onnx")
