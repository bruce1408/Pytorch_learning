import os
import torch
import random
import torch.nn as nn
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class CustomData(data.Dataset):
    def __init__(self, imgFolder, transform=None, train=True, val=False, test=False, splitnum=0.8):
        self.train = train
        self.val = val
        self.test = test
        self.transform = transform

        imgs = [os.path.join(imgFolder, imgFile) for imgFile in os.listdir(imgFolder)]
        self.imgnum = len(imgs)
        self.imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        if train:
            self.imgs = imgs[:int(splitnum * self.imgnum)]
        elif val:
            self.imgs = imgs[int(splitnum * self.imgnum):]
        else:
            self.imgs = imgs
        random.shuffle(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        imgdata = Image.open(img_path)
        if self.transform is not None:
            imgdata = self.transform(imgdata)
        return imgdata, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    data = CustomData('/raid/bruce/datasets/dogs_cats/train')
    print(data[0])
