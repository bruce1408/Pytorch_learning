import cv2
import numpy as np
# b1 = np.fromfile("/Users/bruce/Downloads/dra/data/0.bin", dtype=np.uint8)
# # 3*576*1024 = 1769472
# print("the first bin file size: ",b1.shape)

# # 3*288*512
# b2 = np.fromfile("/Users/bruce/Downloads/dra/ext_data0/0.bin", dtype=np.uint8)
# print("the second bin file size", b2.shape)

# b3 = np.fromfile("/Users/bruce/PycharmProjects/Pytorch_learning/imagenet_224.bin", dtype=np.uint8)
# print(b3.shape)



b4 = np.fromfile("/Users/bruce/Downloads/1.5_Ti_model_files/output_infer.bin")
print(b4.shape)
print(np.argmax(b4))