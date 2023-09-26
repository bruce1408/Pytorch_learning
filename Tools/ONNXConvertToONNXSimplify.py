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
    onnx_simplify("/Users/bruce/Downloads/5223_bev_trans/20230925/modified_0925_mtn_without_linear_dummy_v5.onnx")
