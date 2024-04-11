# import torch
import os
import onnx
import onnxruntime as ort
from onnxsim import simplify
from printk import print_colored_box, print_colored_box_line 


def onnx_simplify(path):
    # simplify the onnx model & load onnx model and check if it is valid
    onnx.load(path)
    onnx_model = onnx.load(path)  
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    
    output_path = path.split(".")[0] + "_onnxsim." + path.split(".")[-1]
    
    onnx.save(model_simp, output_path)
    
    print_colored_box(f"The simplified ONNX model is saved in {output_path}")


def onnxruntime_inference(onnx_model_path):
    # 加载ONNX模型
    ort_session = ort.InferenceSession(onnx_model_path)

    for input_tensor in ort_session.get_inputs():
        # 获取输入的名称、形状和数据类型
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        print_colored_box(f"input name is {input_name}, Shape: {input_shape}, type is {input_tensor.type}", 50)


def print_onnx_input_output(model_path):
    # 加载ONNX模型
    model = onnx.load(model_path)

    # 打印模型的输入信息
    print("Model Inputs:")
    for input in model.graph.input:
        # 打印输入的形状和类型
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        data_type = input.type.tensor_type.elem_type
        
        # print_colored_box(f"input name is {input.name}, Shape: {shape}, type is {onnx.TensorProto.DataType.Name(data_type)}", 50)
        print(input.name, end=': ')

    # 打印模型的输出信息
    print("\nModel Outputs:")
    for output in model.graph.output:
        print(output.name, end=': ')
        # 打印输出的形状和类型
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        data_type = output.type.tensor_type.elem_type
        print("Shape:", shape, "Type:", onnx.TensorProto.DataType.Name(data_type))

if __name__=="__main__":
    # model_path = "/Users/bruce/Downloads/8620_deploy/swin_tiny_patch4_window7_224_224_elementwise_affine.onnx"
    # model_path = "/Users/bruce/Downloads/obstacle_v1.0.onnx"
    # model_path = "/mnt/share_disk/bruce_cui/onnx_models/laneline_154w_20240320_fastbev.onnx"
    model_path = "/mnt/share_disk/bruce_cui/onnx_models/laneline_20240330_fastbev_wo_argmax.onnx"
    # model_path = "/Users/bruce/Downloads/8620_deploy/Laneline/models/epoch_latest_0302.onnx"
    
    onnx_simplify(model_path)
    # print_onnx_input_output(model_path)
    # onnxruntime_inference(model_path)
    
