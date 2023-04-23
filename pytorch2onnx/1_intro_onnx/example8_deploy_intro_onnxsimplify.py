import torch
import onnx
import torch.nn as nn
from onnxsim import simplify


def onnx_simplify(path):
    # simplify the onnx model & load onnx model
    onnx_model = onnx.load(path)  
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    
    output_path = "official_layernorm_onnsim.onnx"
    
    onnx.save(model_simp, output_path)
    
    print("The simplified ONNX model is saved to {}".format(output_path))

if __name__ == '__main__':
    # path = "/home/cuidongdong/Pytorch_learning/Deploy/custom_operator_complicated/pytorch_custom_op/model_complicate.onnx"
    # path = "/home/cuidongdong/onnxruntime_deploy/custom_operator/custom_operator_layernorm/model_layernorm.onnx"
    # path = "custom_operator/custom_operator_layernorm/model_layernorm_v3.onnx"
    # path = "/home/cuidongdong/onnxruntime_deploy/model_resnet18.onnx"
    path = "/home/cuidongdong/onnxruntime_deploy/official_layernorm.onnx"

    onnx_simplify(path)