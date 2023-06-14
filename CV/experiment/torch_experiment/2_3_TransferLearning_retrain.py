import os
import time

import torch
import torch.nn as nn
import random
from torchvision.datasets import FakeData
from PIL import Image
import torch.utils.data as data
import argparse
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
from apex import amp
from apex.parallel import DistributedDataParallel

"""
https://cloud.tencent.com/developer/article/1435646
"""
# parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
batchsize = 64
num_workers = 4


class CustomData(data.Dataset):
    def __init__(self, root, transform=None, train=True, val=False):
        self.val = val
        self.train = train
        self.transform = transform
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        if self.val:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            # 根据图片的num排序，如 cat.11.jpg -> 11
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))  # 所有图片排序
        # imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))  # 所有图片排序

        imgs_num = len(imgs)
        if self.train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        random.shuffle(imgs)  # 打乱顺序

    # 作为迭代器必须有的方法
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0  # 狗的label设为1，猫的设为0
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


# 对数据集训练集的处理，其实可以直接放到 DogCat 类里面去
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),  # 先调整图片大小至256x256
    transforms.RandomCrop((224, 224)),  # 再随机裁剪到224x224
    transforms.RandomHorizontalFlip(),  # 随机的图像水平翻转，通俗讲就是图像的左右对调
    transforms.ToTensor(),  # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))  # 归一化，数值是用ImageNet给出的数值
])

# 对数据集验证集的处理
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 生成训练集和验证集

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--dummy", default=True, type=bool, help="use toy data")
args = parser.parse_args()

if args.dummy:
    trainset = FakeData(10000, (3, 224, 224), 2, transforms.ToTensor())
    valset = FakeData(2000, (3, 224, 224), 2, transforms.ToTensor())
else:
    trainset = CustomData('/datasets/cdd_data/dogs_cats/train', transform=transform_train)
    valset = CustomData('/datasets/cdd_data/dogs_cats/train', transform=transform_val, train=False, val=True)
# 将训练集和验证集放到 DataLoader 中去，shuffle 进行打乱顺序（在多个 epoch 的情况下）
# num_workers 加载数据用多少的子线程（windows不能用这个参数）
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=False, num_workers=num_workers)


class Net(nn.Module):
    def __init__(self, model):
        """
        定义一个Net网络,封装之前的model,然后去掉最后一层,修改最后一层的值为分类数目即可.
        :param model:
        """
        super(Net, self).__init__()
        # 去掉model的最后1层
        # print(model)
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        # print('self resnet layers: \n', self.resnet_layer)
        self.Linear_layer = nn.Linear(512, 2)  # 加上一层参数修改好的全连接层

    def forward(self, x):
        x = self.resnet_layer(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    for batch_idx, (img, label) in enumerate(trainloader):  # 迭代器，一次迭代 batch_size 个数据进去
        optimizer.zero_grad()
        image = img.to(device)
        label = label.to(device)
        out = model(image)
        loss = criterion(out, label)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()
        train_acc = get_acc(out, label)
        print("Epoch:%d [%d|%d] loss:%f acc:%f" % (epoch, batch_idx, len(trainloader), loss.mean(), train_acc))


def val(epoch):
    print(5*"="+"Validation Epoch: %d" % epoch)
    print(len(valloader))
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(valloader):
            image = img.to(device)
            label = label.to(device)
            out = model(image)

            _, predicted = torch.max(out.data, 1)

            total += image.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
            print("Epoch:%d [%d|%d] total:%d correct:%d" % (epoch, batch_idx, len(valloader), total, correct.numpy()))
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


if __name__ == '__main__':
    # 修改最后一全连接层输出维度，但是参数全部要更新训练
    resnet = resnet18(pretrained=False, progress=True)  # 直接用 resnet 在 ImageNet 上训练好的参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若能使用cuda，则使用cuda

    # 修改全连接层
    model = Net(resnet)

    # 放到 GPU 上跑
    model = model.to(device)

    # 设置优化器训练细节
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # apex 加速训练
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss()
    begin_time = time.time()
    for epoch in range(20):
        train(epoch)
        print("epoch time: ", time.time() - begin_time)
        val(epoch)
    torch.save(model, 'modelcatdog.pt')  # 保存模型
    print("train model use time: ", time.time() - begin_time)
