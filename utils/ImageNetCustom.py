import os
import cv2
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from collections import defaultdict
import torchvision.transforms as transforms

# 默认输入网络的图片大小
IMAGE_SIZE = 224

# 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式
dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # 将图像按比例缩放至合适尺寸
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 从图像中心裁剪合适大小的图像
    transforms.ToTensor()  # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]，同时将H×W×C的数据转置成C×H×W，这一点很关键
])


class ImageNetCustom(data.Dataset):  # 新建一个数据集类，并且需要继承PyTorch中的data.Dataset父类
    def __init__(self, mode, dir, dataTransform=dataTransform):  # 默认构造函数，传入数据集类别（训练或测试），以及数据集路径
        self.mode = mode
        self.list_img = []  # 新建一个image list，用于存放图片路径，注意是图片路径
        self.list_label = []  # 新建一个label list，用于存放图片对应猫或狗的标签，其中数值0表示猫，1表示狗
        self.data_size = 0  # 记录数据集大小
        self.transform = dataTransform  # 转换关系
        self.dir = dir
        self.label2category = defaultdict(list)
        self.idx2label = dict()
        self.label2idx = dict()

        self.get_label_map()
        if self.mode == "train":
            dir = os.path.join(os.path.join(dir, "ILSVRC/Data/CLS-LOC"), self.mode)
            for file in os.listdir(dir):  # 遍历dir文件夹
                for imgpath in os.listdir(os.path.join(dir, file)):
                    self.list_img.append(os.path.join(os.path.join(dir, file), imgpath))  # 将图片路径和文件名添加至image list
                    self.data_size += 1  # 数据集增1
                    name = imgpath.split(sep='_')[0]
                    self.list_label.append(self.label2idx[name])
        elif self.mode == "val":
            dir = os.path.join(os.path.join(dir, "ILSVRC/Data/CLS-LOC"), self.mode)
            for imgpath in os.listdir(dir):
                self.list_img.append(os.path.join(dir, imgpath))  # 将图片路径和文件名添加至image list
                self.data_size += 1  # 数据集增1
                name = imgpath.split(sep='_')[0]
                self.list_label.append(self.label2idx[name])

    def __getitem__(self, item):  # 重载data.Dataset父类方法，获取数据集中数据内容
        img = Image.open(self.list_img[item])  # 打开图片
        if img.mode != 'L':
            gray_pic = img.convert("L")
            rgb_pic = gray_pic.convert("RGB")
        else:
            rgb_pic = img.convert('RGB')
            # 将数组转换为图像
        img = Image.fromarray(np.asarray(rgb_pic))
        # print(img.mode)
        label = self.list_label[item]  # 获取image对应的label
        return self.transform(img), torch.LongTensor([label])  # 将image和label转换成PyTorch形式并返回

    def __len__(self):
        return self.data_size  # 返回数据集大小

    def get_label_map(self):
        # n01558993 robin, American robin, Turdus migratorius
        count = 0
        with open(os.path.join(self.dir, "LOC_synset_mapping.txt")) as f:
            for eachlabel in f:
                line_list = eachlabel.strip("\n").split(",")
                label = line_list[0].split(" ")[0]
                self.idx2label[count] = label
                count += 1
                self.label2category[label].append(line_list[0].split(" ")[1])
                for index in range(1, len(line_list)):
                    self.label2category[label].append(line_list[index].strip(" "))
        self.label2idx = {value: key for key, value in self.idx2label.items()}
        return self.label2category


if __name__ == "__main__":
    path = "/data/cdd_data/imagenet_data"
    "/data/cdd_data/imagenet_data/ILSVRC/Data/CLS-LOC"
    # data = ImageNetCustom("train", path, dataTransform=dataTransform)
    # print(data.label2idx)
    # print(data.list_label.__len__())
    # print(data.list_img)
    # print(data.label2idx)
    # for i in range(len(data)):
    #     print(data[0][0].shape, data[0][1])
    # print(data.label2idx["n01558993"])
    # print(data.idx2label[15])
    train_data = ImageNetCustom("train", path)
    print(train_data[0][0].shape)
    train_size = int(0.8 * len(train_data))
    test_size = len(train_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

    for data in train_dataset:
        if data[0].shape[0] == 1:
            print(1)
        elif data[0].shape[0] == 4:
            print(4)
        else:
            pass

