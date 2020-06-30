import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import numpy as np
from torchsummary import summary
import os
# from model.resnet import resnet101
# from dataset.Custom import CustomData
from utils.DataSet_train_val_test import CustomData
# from utils.Custom import CustomData


# parameters
gamma = 0.96
num_workers = 2
batchsize = 128
epochs = 2000
learning_rate = 0.001
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

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
valset = CustomData('/raid/bruce/datasets/dogs_cats/train', transform=transform_val, train=False, val=True, test=False, splitnum=0.8)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=False, num_workers=num_workers)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 14
        )
        self.fc1 = nn.Linear(256*14*14, 1024)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, input):
        output = self.net(input)
        output = output.view(output.shape[0], -1)
        output = self.fc1(output)
        output = self.drop(output)
        output = self.fc2(output)
        output = self.drop(output)
        output = self.fc3(output)
        return output


# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     net = Net()
#     summary(net, (3, 224, 224))

model = Net()
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

criterion = nn.CrossEntropyLoss()
criterion.cuda()


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
        print("Epoch:%d [%d|%d] loss:%f, lr:%f" % (epoch, batch_idx, len(trainloader), loss.mean(), scheduler.get_lr()[0]))


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
            print("Epoch:%d [%d|%d] total:%d correct:%d" % (epoch, batch_idx, len(valloader), total, correct.numpy()))
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


for epoch in range(epochs):

    train(epoch)
    val(epoch)
    # if epoch % 20 == 0 and epoch is not 0:
    #     val(epoch)
torch.save(model.state_dict(), 'ckp/model.pth')
