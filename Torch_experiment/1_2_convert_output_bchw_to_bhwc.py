import onnxruntime as ort
import numpy as np

def load_and_run_onnx_model(model_path, input_data_list):
    """
    Load an ONNX model and run inference.

    Args:
    model_path (str): Path to the ONNX model file.
    input_data (np.ndarray): Input data in BCHW format.

    Returns:
    np.ndarray: Model output in BHWC format.
    """
    sess = ort.InferenceSession(model_path)

    # Get input names from the model
    input_names = [input.name for input in sess.get_inputs()]

    # Ensure the number of inputs matches
    if len(input_data_list) != len(input_names):
        raise ValueError("Number of inputs provided does not match model's requirement.")

    # Prepare input dict
    input_dict = {name: data for name, data in zip(input_names, input_data_list)}

    # Run the model
    outputs = sess.run(None, input_dict)
    

    # Assume the output is the first output, convert it from BCHW to BHWC
    for output in outputs:
        output_bhwc = convert_bchw_to_bhwc(output)

    return output_bhwc


def convert_bchw_to_bhwc(tensor):
    """
    Convert a tensor from BCHW format to BHWC format.
    """
    return np.transpose(tensor, (0, 2, 3, 1))

# Example usage
# Create dummy input data that matches the model's input shape requirements
# For example, a model that expects 3 channel image data, size 256x256, with a batch size of 1


input_data1 = np.random.rand(1, 64, 72, 128).astype(np.float32)
input_data2 = np.random.rand(1, 64, 72, 128).astype(np.float32)
input_data3 = np.random.rand(1, 64, 72, 128).astype(np.float32)
input_data4 = np.random.rand(1, 64, 72, 128).astype(np.float32)
input_data5 = np.random.rand(1, 64, 72, 128).astype(np.float32)
input_data6 = np.random.rand(1, 64, 40, 64).astype(np.float32)
input_data7 = np.random.rand(1, 64, 72, 128).astype(np.float32)
input_data8 = np.random.rand(1, 64, 72, 128).astype(np.float32)
input_data9 = np.random.rand(4, 80, 80, 2).astype(np.float32)
input_data10 = np.random.rand(4, 80, 80, 2).astype(np.float32)
input_data11 = np.random.rand(4, 80, 40, 2).astype(np.float32)
input_data12 = np.random.rand(4, 80, 40, 2).astype(np.float32)
input_data13 = np.random.rand(4, 40, 80, 2).astype(np.float32)
input_data14 = np.random.rand(4, 80, 80, 2).astype(np.float32)
input_data15 = np.random.rand(4, 40, 40, 2).astype(np.float32)
input_data16 = np.random.rand(4, 40, 40, 2).astype(np.float32)
input_data_list = [input_data1, input_data2, input_data3, input_data4, input_data5, input_data6, input_data7, input_data8, input_data9, input_data10, input_data11, input_data12, input_data13, input_data14, input_data15, input_data16]


# Path to your ONNX model
model_path = '/mnt/share_disk/bruce_cui/onnx_models/hm_hp370_bev_v2.6_op16_vt_obstacle_onnxsim.onnx'

# Load the model, run inference
model_outputs = load_and_run_onnx_model(model_path, input_data_list)

# Process the outputs as needed
for output in model_outputs:
    print("Output shape:", output.shape)
    # If you need to convert output format, call the conversion function

