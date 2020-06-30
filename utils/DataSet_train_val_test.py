import os
import torch
import random
import torch.nn as nn
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class CustomData(data.Dataset):
    def __init__(self, root, transform=None, train=True, val=False):
        self.train = train
        self.val = val
        self.test = test
        self.transform = transform
        imgs = [os.path.join(root, imgFile) for imgFile in os.listdir(root)]
        self.imgnum = len(imgs)
        self.imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        if self.test:
            self.imgs = imgs
        else:  # 训练集的话还要还分验证集和测试集
            random.shuffle(imgs)
            if self.train:
                self.imgs = imgs[:int(0.7 * self.imgnum)]
            else:
                self.imgs = imgs[int(0.7 * self.imgnum):]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        imgdata = Image.open(img_path)
        imgdata = self.transform(imgdata)
        return imgdata, label

    def __len__(self):
        return self.imgnum


if __name__ =='__main__':
    CustomData('/raid/bruce/datasets/dogs_cats/train')


