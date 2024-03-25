import onnx 
import numpy as np
from onnx import numpy_helper
from printk import print_colored_box, print_colored_box_line


def expand_dim(model_path):
    # 加载原始模型
    onnx_model = onnx.load(model_path)

    # 获取模型的输出张量
    output_tensor = None
    for node in onnx_model.graph.output:
        if node.name == "output":  # 将 "output" 替换模型中实际的输出张量名称
            output_tensor = node
            break
    print(output_tensor)
    # 修改输出张量的维度
    if output_tensor is not None:
        output_tensor.type.tensor_type.shape.dim.insert(0, onnx.TensorShapeProto.Dimension(dim_value=1))

    # 保存修改后的模型
    onnx.save(onnx_model, "./modified_model.onnx")
    print_colored_box("The modified ONNX model is saved in ./modified_model.onnx")

        
if __name__ == "__main__":
    model_path="/Users/bruce/Downloads/Chip_test_models/backbone_224_224/regnety_002.onnx"
    expand_dim(model_path)