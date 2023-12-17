import onnx
import numpy as np
import onnxruntime as rt
import onnx.helper as helper
import onnx.numpy_helper as np_helper

def reshape_input(onnx_model, input_name, in_shape):
    idx = 0
    dtype = 0
    for it in range(len(onnx_model.graph.input)):
        if onnx_model.graph.input[it].name == input_name:
            dtype = onnx_model.graph.input[it].type.tensor_type.elem_type
            onnx_model.graph.input.remove(onnx_model.graph.input[it])
            idx = it
            break
    input_node = helper.make_tensor_value_info(input_name, dtype, in_shape)
    onnx_model.graph.input.insert(idx, input_node)

def reshape_output(onnx_model, output_name, out_shape):
    idx = 0
    dtype = 0
    for it in range(len(onnx_model.graph.output)):
        if onnx_model.graph.output[it].name == output_name:
            dtype = onnx_model.graph.output[it].type.tensor_type.elem_type
            onnx_model.graph.output.remove(onnx_model.graph.output[it])
            idx = it
            break
    output_node = helper.make_tensor_value_info(output_name, dtype, out_shape)
    onnx_model.graph.output.insert(idx, output_node)

if __name__ == "__main__":
    onnx_model_path = "./models/mobilenetv2-576-1024.onnx"
    onnx_model_save_path = "./models/mobilenetv2_224_224.onnx"

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    reshape_input(onnx_model, "input", [1,3,224,224])
    reshape_output(onnx_model, "output", [1,1000])
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_model_save_path)