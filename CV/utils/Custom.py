import os
import random
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms


class CustomData(data.Dataset):
    def __init__(self, root, transform=None, train=True, test=False):
        self.test = test
        self.train = train
        self.transform = transform
        imgs = [os.path.join(root, img) for img in os.listdir(root)]  # img path

        # test1: Dataset/test1/8973.jpg
        # train: Dataset/train/cat.10004.jpg

        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))  # 图片排序

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        else:  # 训练集的话还要还分验证集和测试集
            random.shuffle(imgs)
            if self.train:
                self.imgs = imgs[:int(0.7 * imgs_num)]
            else:
                self.imgs = imgs[int(0.7 * imgs_num):]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    trainset = CustomData('../../Dataset/dogs_cats/train', transform=None)
