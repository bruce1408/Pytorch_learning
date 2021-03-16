import torchvision
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

img_path = "../../Dataset/dogs_cats/train/dog.9.jpg"

# 引入transforms.ToTensor()功能： range [0, 255] -> [0.0, 1.0]
transform1 = transforms.Compose([transforms.ToTensor()])

# 直接读取：numpy.ndarray
img = cv2.imread(img_path)
print("img.shape = ", img.shape)

# 归一化，转化为numpy.ndarray并显示
img1 = transform1(img)
img2 = img1.numpy() * 255
img2 = img2.astype('uint8')
img2 = np.transpose(img2, (1, 2, 0))

print("img1 = ", img1)
# cv2.imshow('img2 ', img2)
# cv2.waitKey(0)

# PIL 读取图像
img = Image.open(img_path).convert('RGB')  # 读取图像
img2 = transform1(img)  # 归一化到 [0.0,1.0]
print("img2 = ", img2)  # 转化为PILImage并显示
img_2 = transforms.ToPILImage()(img2).convert('RGB')
print("img_2 = ", img_2)
# img_2.show()

import os
print(os.path.dirname(__file__))
print(os.path.abspath(os.path.join(os.getcwd(), "../../")))