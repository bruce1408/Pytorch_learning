import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import numpy as np
import os
from model.resnet import resnet101
from dataset.Custom import CustomData

# parameters
num_workers = 2
batchsize = 64
epochs = 20
lr = 0.001
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = CustomData('/raid/bruce/datasets/dogs_cats/train', transform=transform_train)
valset = CustomData('/raid/bruce/datasets/dogs_cats/train', transform=transform_val)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=False, num_workers=num_workers)

model = resnet101(pretrained=True)
model.fc = nn.Linear(2048, 2)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=3)
criterion = nn.CrossEntropyLoss()
criterion.cuda()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    model.train()
    for batch_idx, (img, label) in enumerate(trainloader):
        image = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        print("Epoch:%d [%d|%d] loss:%f" % (epoch, batch_idx, len(trainloader), loss.mean()))


def val(epoch):
    print("\nValidation Epoch: %d" % epoch)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(valloader):
            image = img.cuda()
            label = label.cuda()
            out = model(image)
            _, predicted = torch.max(out.data, 1)
            total += image.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


for epoch in range(epochs):
    train(epoch)
    val(epoch)
torch.save(model.state_dict(), 'ckp/model.pth')
