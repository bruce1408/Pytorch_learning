# import torch
# # import os
import onnx
from onnxsim import simplify


def onnx_simplify(path):
    # simplify the onnx model & load onnx model
    onnx_model = onnx.load(path)  
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    
    output_path = path.split(".")[0] + "_onnxsim." + path.split(".")[-1]
    
    onnx.save(model_simp, output_path)
    
    print("The simplified ONNX model is saved to {}".format(output_path))


if __name__=="__main__":
    onnx_simplify("/Users/bruce/Chip_test/Cambricon_chip_test_results/onnx_yolop/yolop-320-320.onnx")
