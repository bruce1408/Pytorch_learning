import os
import torch.utils.data as data
from PIL import Image
import torch
import sys
sys.path.append("../..")
import argparse
import torch.nn as nn
from utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
# from mmdet.models.backbones.efficientnet import EfficientNet
from mmdet.models.backbones.mobilenet_v2 import MobileNetV2
from mmdet.models.backbones.resnet import ResNet
from utils.ImageNetCustom import ImageNetCustom
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed


# 默认输入网络的图片大小
IMAGE_SIZE = 224
epochs = 5000
# learning_rate = 1e-4
learning_rate = 1e-2

totalCount = 0
# 图片处理
dataTransform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 尺寸变化
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 中心裁剪
    transforms.ToTensor()  # 归一化
])

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # 模型选择
    parser.add_argument('--model_name', default="resnet-50", help='model name resnet-50、mobilenet_v2、efficientnet')
    # parser.add_argument('--config', default="../configs/bevdet/bevdet_mobilenetv2.py", help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    # parser.add_argument("--log_dir", help="view loss")

    parser.add_argument('--load_from', default="",
                        help='laod checkpoints from saved models')

    parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5], help="gpu设备编号")
    parser.add_argument('--num_workers', type=int, default=6, help='')
    parser.add_argument('--batch_size', type=int, default=6, help='')
    parser.add_argument('--gpu', default=6, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='GPU通信方式用nccl')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='总共的进程数目')
    # parser.add_argument('--distributed', action='store_true', help='')
    args = parser.parse_args()

    return args


args = parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
log = Logger('imagenet_%s.log' % args.model_name, level='info')

# class CustomData(data.Dataset):
#     def __init__(self, mode, imgpath):
#         self.mode = mode
#         self.list_img = []
#         self.list_label = []
#         self.imgMode = {}
#         self.data_size = 0
#         self.transform = dataTransform
#         # label到num的转换
#         self.label2num = {w: i for i, w in enumerate(sorted(os.listdir(os.path.join(imgpath, mode))))}
#
#         imgFolder = os.path.join(imgpath, self.mode)
#         for file in os.listdir(imgFolder):
#             each_catorgy_dir = os.path.join(imgFolder, file)
#             for img_path in os.listdir(each_catorgy_dir):
#                 self.list_img.append(os.path.join(each_catorgy_dir, img_path) + '&' + file)
#                 self.data_size += 1
#
#     def __getitem__(self, item):
#         img = self.list_img[item].split('&')[0]
#         label = self.label2num[self.list_img[item].split("&")[1]]
#         img = Image.open(img)
#         return self.transform(img), torch.tensor([label], dtype=torch.long)
#
#     def __len__(self):
#         return self.data_size

class Net(nn.Module):
    def __init__(self, model_name, model, labels=1000):
        super(Net, self).__init__()
        self.model = model
        self.label_size = labels
        if model_name == "resnet":  # 2048, 7, 7
            self.fc = nn.Linear(2048*7*7, self.label_size)
        elif model_name == "mobilenet_v2":
            self.fc = nn.Linear(1280*7*7, self.label_size)  # 1280, 7, 7
        elif model_name == "efficientnet":
            self.fc = nn.Linear(1536*7*7, self.label_size)  # efficientNet-b3 output size = 1536, 7, 7

    def forward(self, x):
        output = self.model(x)[0]
        # print(output.__len__())
        B, C, H, W = output.shape
        output = self.fc(output.view(B, -1))
        return output


def get_acc(pred, label):
    total = pred.shape[0]
    _, pred_label = pred.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total


def train(model, trainloader, epoch, lr_schedule, optimizer, writer, criterion, device):
    model.train()
    global totalCount
    for index, (img, label) in enumerate(trainloader):
        img = img.to(device)
        label = label.squeeze(1).to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # print('image shape: ', img.shape)
        out = model(img)
        # print("outshpa", out.shape, label.shape)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        # if index % 100 == 0 and index is not 0:
        # lr_schedule.step()
        train_acc = get_acc(out, label)
        totalCount += 1
        writer.add_scalar("loss", loss, totalCount)
        # log.logger.debug('debug')
        if dist.get_rank() == 0:
            log.logger.info("text: Epoch:%d [%d|%d] loss:%f acc:%f, lr:%f" % (epoch, index, len(trainloader),
                                                                    loss.mean(), train_acc, lr_schedule.get_last_lr()[0]))


def val(model, valloader, epoch, device):
    print('begin to eval')
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for index, (img, label) in enumerate(valloader):
            img = img.to(device)
            label = label.squeeze(1).to(device)
            out = model(img)
            _, pred = torch.max(out.data, 1)
            total += img.shape[0]
            correct += pred.data.eq(label.data).cpu().sum()
            # print("Epoch:%d [%d|%d] total:%d correct:%d" % (epoch, index, len(valloader), total, correct.numpy()))
    # print("Acc: %f " % ((1.0 * correct.numpy()) / total))
    if dist.get_rank() == 0:
        log.logger.info("Acc: %f " % ((1.0 * correct.numpy()) / total))

def main_worker(gpu, ngpus_per_node, args):
    # global model

    args.gpu = gpu
    # ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(args.gpu))
    # print(args.gpu, args.rank)

    args.rank = args.rank * ngpus_per_node + gpu

    torch.cuda.set_device(args.gpu)  # 这句话放哪里都一样啊
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    # torch.cuda.set_device(args.gpu)

    # 每张卡上多少batch_size进行训练
    args.batch_size = int(args.batch_size / ngpus_per_node)

    # 数据处理works
    args.num_workers = int(args.num_workers / ngpus_per_node)

    path = "/data/cdd_data/imagenet_data"
    writer = SummaryWriter(args.model_name + '_logs')

    train_data = ImageNetCustom("train", path, dataTransform)
    train_size = int(0.8 * len(train_data))
    test_size = len(train_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers,
                                              sampler=train_sampler)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers,
                                            sampler=val_sampler)

    # 定义模型
    if args.model_name == "efficientnet":
        # checkpoint = '/home/cuidongdong/backbone_evaluation/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa
        checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/' \
                     'efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'
        model_sampel = EfficientNet(
            arch='b3',
            out_indices=[6],
            init_cfg=dict(
                type='Pretrained', checkpoint="torchvision://efficientnet"))
        model = Net("efficientnet", model_sampel)

    elif args.model_name == "resnet-50":
        model_sampel = ResNet(
            depth=50,
            num_stages=4,
            out_indices=([3]),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            with_cp=True,
            init_cfg=dict(
                type='Pretrained', checkpoint="")
        )
        "/home/cuidongdong/backbone_evaluation/resnet50-0676ba61.pth"
        model = Net("resnet", model_sampel)
    elif args.model_name == "resnet-101":
        model_sampel = ResNet(
            depth=101,
            num_stages=4,
            out_indices=([3]),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            with_cp=True,
            init_cfg=dict(
                type='Pretrained', checkpoint="/home/cuidongdong/backbone_evaluation/resnest101_d2-f3b931b2.pth")
        )
        model = Net("resnet", model_sampel)
    elif args.model_name == "mobilenet_v2":
        model_sampel = MobileNetV2(
            # arch='b3',
            out_indices=[7],
            init_cfg=dict(
                type='Pretrained', checkpoint="torchvision://mobilenetv2"))
        model = Net("mobilenet_v2", model_sampel)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs to train the model with Distributed!")
        # model = nn.DataParallel(model)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # if torch.cuda.is_available():
    # model.cuda()
    # model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.9)
    # lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3, 5, 10, 20, 35], gamma=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        # train(model, epoch, lr)
        train(model, trainloader, epoch, lr_schedule, optimizer, writer, criterion, args.gpu)
        if dist.get_rank() == 0:
            val(model, valloader, epoch, args.gpu)
            torch.save(model, args.model_name + '_2backbone.pth')
        lr_schedule.step()


def main():
    args = parse_args()
    # os.makedirs(args.output, exist_ok=True)

    # 这里这种写法失效了，因为这里调用的包mmcv里面的模型默认使用全部gpu资源
    # ngpus_per_node = torch.cuda.device_count()

    # 显式的设置gpu资源卡数
    ngpus_per_node = 6
    # print("has gpu nums: ", ngpus_per_node)

    # world_size表示总进程数，一般几张卡就开几个进程
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


if __name__ == "__main__":
    main()


    # input = torch.randn(2, 3, 224, 224)
    # efficient_model = EfficientNet(arch='b3', out_indices=(4, 5, 6))
    # output = efficient_model(input)
    # print('eff', output[0].shape)
    # print('eff', output[1].shape)
    # print('eff', output[2].shape)
    # summary(efficient_model.cuda(), (3, 224, 224))
    #
    # mobilenet = MobileNetV2(out_indices=(4, 7))  # range 0-7,共8层
    # output1 = mobilenet(input)
    # print(output1.__len__())
    # print('mobil', output1[0].shape)
    # print('mobil', output1[1].shape)
    #
    # resnet = ResNet(depth=50, out_indices=(2, 3))
    # output2 = resnet(input)
    # print('resnet', output2[0].shape)
    # print('resnet', output2[1].shape)
    #
    # resnet = ResNet(depth=101, out_indices=(2, 3))
    # output2 = resnet(input)
    # print('resnet', output2[0].shape)
    # print('resnet', output2[1].shape)
