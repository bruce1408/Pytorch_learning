import onnx
import onnxruntime as ort
import numpy as np

# Load ONNX model
model_path = "/Users/bruce/Downloads/5223_bev_trans/20230821/0821_mtn_dummy_lane_edge_obstacle_split_input_norm.onnx"
sess = ort.InferenceSession(model_path)

# Get input and output names and shapes
input_name = sess.get_inputs()[0].name
outputs = sess.get_outputs()
print("=" * 50)
for input_name in (sess.get_inputs()):
    print(input_name.name)
    
for output_name in sess.get_outputs():
    # print(sess.get_outputs())
    print(output_name.name)

# 获取name
# output_name1 = sess.get_outputs()[0].name
# output_name2 = sess.get_outputs()[1].name
# output_name3 = sess.get_outputs()[2].name

# # 获取shape
# input_shape = sess.get_inputs()[0].shape
# output_shape1 = sess.get_outputs()[0].shape
# output_shape2 = sess.get_outputs()[1].shape
# output_shape3 = sess.get_outputs()[2].shape
# print(output_name1, output_name2, output_name3)


# # Change input and output shapes to (1, ...)
# input_shape[0] = 1

# output_shape1[0] = 1
# output_shape2[0] = 1
# output_shape3[0] = 1

# # Update model with new input and output shapes
# model = onnx.load(model_path)
# model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
# model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
# model.graph.output[1].type.tensor_type.shape.dim[0].dim_value = 1
# model.graph.output[2].type.tensor_type.shape.dim[0].dim_value = 1
# onnx.save(model, "/Users/bruce/Downloads/amb_models/8tflops_cut_reshape_batchsize11.onnx")

# # Prepare input data
# input_data = np.zeros(tuple(input_shape), dtype=np.float32)

# # Perform inference
# # output = sess.run([output_name1, output_name2, output_name3], {input_name: input_data})

# # # Get output data
# output_data = output[0]
# print(output_data.shape)
