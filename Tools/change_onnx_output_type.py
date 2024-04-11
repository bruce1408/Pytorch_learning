import onnx
from onnx import helper
from printk import print_colored_box

# 加载 ONNX 模型
model_path = '/mnt/share_disk/bruce_cui/onnx_models/laneline_154w_20240320_fastbev_onnxsim.onnx'
model = onnx.load(model_path)

# 假设我们要修改的输出是模型的最后一个输出
# 首先，获取输出的数量
num_outputs = len(model.graph.output)
print(num_outputs)

for output in model.graph.output:

    if output.type.tensor_type.elem_type == onnx.TensorProto.INT64:
        # 修改数据类型为 float32
        output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

# 保存修改后的模型
modified_model_path = '/mnt/share_disk/bruce_cui/onnx_models/laneline_154w_20240320_fastbev_onnxsim_output_float32.onnx'
onnx.save(model, modified_model_path)

print_colored_box("模型输出类型已从 int64 修改为 float32。")

