import torch
import torch.nn as nn
import sys, os
import torch.utils.model_zoo as model_zoo
from torchsummary import summary
from CV.utils.DataSet_train_val_test import CustomData
import torchvision.transforms as transforms
from torchvision.models import resnet50
from CV.models.ResNet_advance import Resnet50
"""
使用官方预训练好的神经网络来加载权重, 自定义一个神经网络B,然后把预训练好的权重赋值给B网络进行训练.
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
save_path = "./transform_resnet50.pt"
gamma = 0.96
num_workers = 4
batchsize = 32  # batch_size 不要太大
epochs = 10
learning_rate = 0.01


mean = [0.485, 0.456, 0.406]
std = [0.2459, 0.2424, 0.2603115]

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop((224, 224), padding=4),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


trainset = CustomData('../../Dataset/dogs_cats/train', transform=transform_train)
valset = CustomData('../../Dataset/dogs_cats/train', transform=transform_val,
                    train=False, val=True, test=False, splitnum=0.8)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=True, num_workers=num_workers)


def get_acc(pred, label):
    total = pred.shape[0]
    _, pred_label = pred.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total


def train(model, epoch):
    model.train()
    optimizer.zero_grad()
    scheduler.step()
    for batch_idx, (img, label) in enumerate(trainloader):
        image = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        acc = get_acc(out, label)
        sys.stdout.write('\033[1;36m \r>>Train Epoch:%d [%d|%d] loss:%f, lr:%f, acc:%f\033[0m' %
                         (epoch, batch_idx, len(trainloader), loss.mean(), scheduler.get_lr()[0], acc))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


def val(model, epoch):
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
            sys.stdout.write('\033[1;35m \r>>Validation Epoch:%d [%d|%d] total:%d, corretc:%d \033[0m' %
                             (epoch, batch_idx, len(valloader), total, correct.numpy()))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


if __name__ == "__main__":
    # ================= 测试网络结构是否正确 =================
    net = Resnet50()
    if torch.cuda.is_available():
        summary(net.cuda(), (3, 224, 224))
        x = torch.rand((1, 3, 224, 224)).cuda()
        output = net(x).cuda()
    else:
        summary(net, (3, 224, 224))
    # =================== 记载预训练模型 ====================
    net = Resnet50()  # 自己定义的网络
    officialNet = resnet50(pretrained=True)  # 官方的网络结构

    # 对官方模型预训练参数进行固定
    for para in officialNet.parameters():
        para.requires_grad = False

    # 更新网络参数到自己定义的网络
    new_state_dict = officialNet.state_dict()
    dd = net.state_dict()
    for index, k in enumerate(new_state_dict.keys()):
        print('total param is: ', k)
        if k in dd.keys() and not k.startswith('fc'):
            print('include param is:', k, index)
            dd[k] = new_state_dict[k]
        else:
            print('='*10, k)
    net.load_state_dict(dd)
    # ==================== 梯度参数设置 ====================
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=0.001, momentum=0.9,
                                weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.96, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    # =================== 模型训练部分 =====================
    for epoch in range(epochs):
        train(net.cuda(), epoch)
        val(net.cuda(), epoch)
        torch.save(net.state_dict(), '6_11_transform.pt')




