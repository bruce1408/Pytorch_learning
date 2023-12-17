import os
import onnx
import onnxruntime
from onnx import shape_inference

path = "/mnt/share_disk/bruce_cui/8tflops_cylinder_freespace.onnx"

# onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)

import os
import onnx
import copy
import numpy as np
import logging
import onnxruntime
from collections import OrderedDict
from onnx import shape_inference
logging.basicConfig(level=logging.INFO)
from onnx import shape_inference, TensorProto, version_converter, numpy_helper
logger = logging.getLogger("[ONNXOPTIMIZER]")

def test_model_by_onnxruntime(model):
    logger.info("Test model by onnxruntime")

    input_shape = model.graph.input[0].type.tensor_type.shape.dim

    image_shape = [x.dim_value for x in input_shape]
    image_shape_new = []
    for x in image_shape:
        if x == 0:
            image_shape_new.append(1)
        else:
            image_shape_new.append(x)
    image_shape = image_shape_new
    img_array = np.array(np.random.random(image_shape), dtype = np.float32)
    img = img_array
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    ort_inputs = {}
    for i, input_ele in enumerate(ort_session.get_inputs()):
        ort_inputs[input_ele.name] = img

    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, ort_inputs)
    ort_outs = OrderedDict(zip(outputs, ort_outs))
    logger.info("Test model by onnxruntime success")
    # del self.model.graph.output[:]
    # model.graph.output.extend(ori_output)
    return ort_outs

onnx_model = onnx.load(path)
ort_outs = test_model_by_onnxruntime(onnx_model)
print(ort_outs.keys())
print(ort_outs['254'].shape)