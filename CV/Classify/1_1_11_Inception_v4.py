import os
import torch
import torch.nn as nn
from CV.utils.dog_cat import DogCat

import torch.utils.data as data
from CV.models.inception_advance import Inception_v1
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# parameters
os.environ['CUDA_VISIBLES_DEVICES'] = '1,2'
batchsize = 32
num_works = 2
epochs = 30
learning_rate = 0.01
gamma = 0.96


trainData = DogCat('../../Dataset/dogs_cats/train')
valData = DogCat("../../Dataset/dogs_cats/train", train=False, test=True)
# testData = CustomData("/raid/bruce/datasets/dogs_cats/train", transform=transform_test, train=False, val=False)

trainloader = torch.utils.data.DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=num_works)
valloader = torch.utils.data.DataLoader(valData, batch_size=batchsize, shuffle=False, num_workers=num_works)


def get_acc(pred, label):
    total = pred.shape[0]
    _, pred_label = pred.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, epoch, lr):
    print("start training the models ")
    model.train()
    lr.step()
    lr_ = lr.get_lr()[0]
    for index, (img, label) in enumerate(trainloader):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_acc = get_acc(out, label)
        print("Epoch:%d [%d|%d] loss:%f acc:%f, lr:%f" % (epoch, index, len(trainloader), loss.mean(), train_acc, lr_))


def val(model, epoch):
    print('begin to eval')
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for index, (img, label) in enumerate(valloader):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            _, pred = torch.max(out.data, 1)
            total += img.shape[0]
            correct += pred.eq(label.data).cpu().sum()
            print("Epoch:%d [%d|%d] total:%d correct:%d" % (epoch, index, len(valloader), total, correct.numpy()))
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


if __name__ == '__main__':
    model = Inception_v1(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.9)
    lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train(model, epoch, lr)
        val(model, epoch)
        lr.step()
    torch.save(model, 'model_cat_dog.pt')
