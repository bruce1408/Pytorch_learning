# # 打开二进制文件以读取
# with open('/Users/bruce/Downloads/5223_bev_trans/output.bin', 'rb') as file:
#     # 读取整个文件内容
#     binary_data = file.read()


# import struct

# # 打开二进制文件以读取
# with open('/Users/bruce/Downloads/15_Ti_model_files/mobilenet_infer_output.bin', 'rb') as file:
#     # 读取4个字节的二进制数据（32位浮点数）
#     binary_data = file.read(4)

# # 解析二进制数据为浮点数
# float_value = struct.unpack('I', binary_data)[0]

# # 打印解析后的浮点数值
# print("Float Value:", float_value)


import os

files = os.listdir("/Users/bruce/Downloads/15_Ti_model_files/calibation_data_bmp")
with open("./bmp_calibration.txt", "w")as f:
    for eachfile in files:
        f.write("/Users/bruce/Downloads/15_Ti_model_files/calibation_data_bmp/" + eachfile + "\n")