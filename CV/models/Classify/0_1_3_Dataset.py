import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as transforms

IMAGE_SIZE = 224

# 图片处理
dataTransform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 尺寸变化
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 中心裁剪
    transforms.ToTensor()  # 归一化
])

class Custom(data.Dataset):
    def __init__(self, mode, imgpath):
        self.mode = mode
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.transform = dataTransform

        if self.mode == 'train':
            imgFolder = os.path.join(imgpath, 'train')
            for file in tqdm(os.listdir(imgFolder)):
                self.list_img.append(os.path.join(imgFolder, file))
                self.data_size += 1
                name = file.split('.')
                if name[0] == 'cat':
                    self.list_label.append(0)
                else:
                    self.list_label.append(1)
        elif self.mode == 'test':
            imgFolder = os.path.join(imgpath, 'test')
            for file in tqdm(os.listdir(imgFolder)):
                self.list_img.append(os.path.join(imgFolder, file))
                self.data_size += 1
                self.list_label.append(2)
        else:
            print('underfined datasets!')
        # print(self.list_img)

    def __getitem__(self, item):
        if self.mode == 'train':
            img = Image.open(self.list_img[item])
            label = self.list_label[item]
            return self.transform(img), torch.Tensor([label])

        elif self.mode == 'test':
            img = Image.open(self.list_img[item])
            return self.transform(img)
        else:
            print('None')

    def __len__(self):
        return self.data_size


if __name__ == '__main__':
    data = Custom('train', '../../Dataset/dogs_cats/')
    print(data.__len__())
    print(data[0])
