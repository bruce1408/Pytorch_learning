import torch
import warnings
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from  torchvision import transforms
from torchvision.datasets import FakeData
warnings.filterwarnings("ignore", category=UserWarning)

from torch.onnx import register_custom_op_symbolic

"""
训练和保存带有torch layernorm算子的模型
"""
def get_acc(pred, label):
    total = pred.shape[0]
    _, pred_label = pred.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 定义模型
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 1, bias=False)  
        self.bn = nn.BatchNorm2d(256)
        self.layernorm = nn.LayerNorm(256)
        self.fc = nn.Linear(25690112, 3)



    def forward(self, x):
        batch_size = x.shape[0]
        output = self.conv1(x)
        output = self.bn(output)
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(1, -1, 256)
        output = self.layernorm(output)
        output = F.interpolate(output, scale_factor=2, mode='nearest')
        output = output.reshape(batch_size, -1)
        output = self.fc(output)
        

        return output

def train(model, epoch, lr):
    print("start training the models ")
    model.train()
    lr_ = lr.get_last_lr()[0]
    for index, (img, label) in enumerate(trainloader):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model(img)
        # print(out.shape, label.shape)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_acc = get_acc(out, label)
        print("Epoch:%d [%d|%d] loss:%f acc:%f, lr:%f" % (epoch, index, len(trainloader), loss.mean(), train_acc, lr_))
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainData = FakeData(10000, (3, 224, 224), 3, transforms.ToTensor())
    valData = FakeData(2000, (3, 224, 224), 3, transforms.ToTensor())
    epochs = 30 
    gamma = 0.96
    num_works = 4
    batchsize = 16
    learning_rate = 0.001
    save_path = "official_layernorm.pth"

    trainloader = torch.utils.data.DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=num_works)
    valloader = torch.utils.data.DataLoader(valData, batch_size=batchsize, shuffle=False, num_workers=num_works)
    
    
    model = CustomModel()
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.9)
    lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train(model, epoch, lr)
        lr.step()
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'lr': lr},
            save_path)
        
    