import os
import sys
import torch
import numpy as np
import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, OrtValueVector, RunOptions

X = torch.randn(3, 2, 1, 2)
num_groups = torch.tensor([2.])
scale = torch.tensor([1., 1.])
bias = torch.tensor([0., 0.])
inputs = (X, num_groups, scale, bias)
    
shared_library = "/home/cuidongdong/Pytorch_learning/Deploy/custom_library_2_without_template/build/bin/libcustomop.so"
if not os.path.exists(shared_library):
    raise FileNotFoundError("Unable to find '{0}'".format(shared_library))

# 定义provider
available_providers_without_tvm_and_tensorrt = [
    provider
    for provider in onnxrt.get_available_providers()
    if provider not in {"TvmExecutionProvider", "TensorrtExecutionProvider"}
]

custom_op_model = "/home/cuidongdong/Pytorch_learning/Deploy/custom_operator_complicated/pytorch_custom_op/model.onnx"
if not os.path.exists(custom_op_model):
    raise FileNotFoundError("Unable to find '{0}'".format(custom_op_model))

so1 = onnxrt.SessionOptions()
so1.register_custom_ops_library(shared_library)

# Model loading successfully indicates that the custom op node could be resolved successfully
sess1 = onnxrt.InferenceSession(
    custom_op_model, sess_options=so1, providers=available_providers_without_tvm_and_tensorrt
)
# Run with input data
input_name_0 = sess1.get_inputs()[0].name
input_name_1 = sess1.get_inputs()[1].name
input_name_2 = sess1.get_inputs()[2].name
input_name_3 = sess1.get_inputs()[3].name

output_name = sess1.get_outputs()[0].name
input_0 = np.ones((3, 5)).astype(np.float32)
input_1 = np.zeros((3, 5)).astype(np.float32)
res = sess1.run([output_name], {input_name_0: X, 
                                input_name_1: num_groups, 
                                input_name_2: scale, 
                                input_name_3:bias})
# output_expected = np.ones((3, 5)).astype(np.float32)
# np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)