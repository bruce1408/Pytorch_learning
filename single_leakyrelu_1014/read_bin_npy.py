import numpy as np

npy_path = "ir_output/debug_info/QuantizedSubGraph_1/Conv_1.npy"
bin_path = "compiler_output/QuantizedSubGraph_1/sim_output/Conv_1.bin"

conv1_data_npy = np.load(npy_path)
print(conv1_data_npy.shape, conv1_data_npy.dtype)
# conv1_data_bin = np.fromfile(bin_path, dtype=conv1_data_npy.dtype).reshape(quant_out.shape)
conv1_data_bin = np.fromfile(bin_path, dtype=conv1_data_npy.dtype).reshape(conv1_data_npy.shape)
print(conv1_data_bin.shape)

print("==============\n",conv1_data_npy)
print("==============\n",conv1_data_bin)
