import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from utils.DataSet_train_val_test import CustomData
from CV.utils.dog_cat import DogCat

import torch.utils.data as data
# from utils.inception_advance import Inception_v1
from CV.utils.VGGNet import VGGNet16

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

# parameters
os.environ['CUDA_VISIBLES_DEVICES'] = '0'
batchsize = 64
num_works = 4
epochs = 30
learning_rate = 0.001
gamma = 0.96
save_path = "./model_cat_dog_lr.pth"

useGpu=False

if torch.cuda.is_available():
    useGpu = True

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
])


transforms_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
])


transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainData = DogCat('/datasets/cdd_data/dogs_cats/train')
valData = DogCat("/datasets/cdd_data/dogs_cats/train", train=False, test=True)
# testData = CustomData("/raid/bruce/datasets/dogs_cats/train", transform=transform_test, train=False, val=False)

trainloader = torch.utils.data.DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=num_works)
valloader = torch.utils.data.DataLoader(valData, batch_size=batchsize, shuffle=False, num_workers=num_works)
# testloader = torch.utils.Dataset.DataLoader(testData, batch_size=batchsize, shuffle=False, num_workers=num_works)


def get_acc(pred, label):
    total = pred.shape[0]
    _, pred_label = pred.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total


# device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')


def train(model, epoch, lr):
    print("start training the models ")
    model.train()
    # lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

    # lr_ = lr.get_lr()[0]
    for index, (img, label) in enumerate(trainloader):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        # if index % 100 == 0 and index is not 0:
        train_acc = get_acc(out, label)
        print("Epoch:%d [%d|%d] loss:%f acc:%f, lr:%f" % (epoch, index, len(trainloader), loss.mean(), train_acc, lr.get_last_lr()[0]))


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
            correct += pred.data.eq(label.data).cpu().sum()
            print("Epoch:%d [%d|%d] total:%d correct:%d" % (epoch, index, len(valloader), total, correct.numpy()))
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


if __name__ == '__main__':
    # 所有参数全部更新
    model = VGGNet16(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # also use cuda: Num
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.9)
    lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3], last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train(model, epoch, lr)
        val(model, epoch)
        lr.step()
    # torch.save(model.state_dict(), save_path)
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'lr': lr},
            save_path)
