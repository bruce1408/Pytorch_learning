import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils.DataSet_train_val_test import CustomData
from utils.Inception_v1 import Inception_v1
import torch.utils.data as data

# parameters
os.environ['CUDA_VISIBLES_DEVICES'] = '1'
batchsize = 32
num_works = 4
epochs = 20

transforms_train = transforms.Compose([
    transforms.Resize((225, 225)),
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

trainData = CustomData('/raid/bruce/datasets/dogs_cats/train', transform=transforms_train)
valData = CustomData("/raid/bruce/datasets/dogs_cats/train", train=False, val=True, transform=transforms_val)
testData = CustomData("/raid/bruce/datasets/dogs_cats/train", train=False, val=False, test=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=num_works)
valloader = torch.utils.data.DataLoader(valData, batch_size=batchsize, shuffle=False, num_workers=num_works)
testloader = torch.utils.data.DataLoader(testData, batch_size=batchsize, shuffle=False, num_workers=num_works)


def get_acc(pred, label):
    total = pred.shape[0]
    _, pred_label = pred.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total


def train(epoch):
    print("start training the models ")
    print(trainloader)
    model.train()
    for index, (img, label) in enumerate(trainloader):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_acc = get_acc(out, label)
        print("Epoch:%d [%d|%d] loss:%f acc:%f" % (epoch, index, len(trainloader), loss.mean(), train_acc))


def val(epoch):
    print(len(valloader))
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
            correct +
if __name__ == '__main__':
    model = Inception_v1(2, mode='train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train(epoch)
        val(epoch)
    torch.save(model, 'model_cat_dog.pt')