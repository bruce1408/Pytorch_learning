import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import logging
from logging import handlers
import sys
sys.path.append("../..")
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import torch.utils.data as data
from PIL import Image
import numpy as np
from collections import defaultdict
from apex import amp
from apex.parallel import DistributedDataParallel

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
#                     help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument("--path", default="/home/cuidongdong/imagenet_data")

parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3467', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=4, type=int,
                    help='GPU id to use. 单卡才会使用')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0, 1, 2, 3], help="gpu设备编号")

parser.add_argument("--work_dir", type=str, default="outputs", help="模型存储路径")

parser.add_argument('--multiprocessing_distributed',
                    # action='store_true',
                    default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training,使用分布式训练')
parser.add_argument('--dummy', default=False, help="use fake data to benchmark")

best_acc1 = 0
totalCount = 0
args = parser.parse_args()
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
os.makedirs(os.path.join(args.arch, args.work_dir), exist_ok=True)


# 默认输入网络的图片大小
IMAGE_SIZE = 224

# 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式
dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # 将图像按比例缩放至合适尺寸
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 从图像中心裁剪合适大小的图像
    transforms.ToTensor()  # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]，同时将H×W×C的数据转置成C×H×W，这一点很关键
])


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.filename = filename
        self.level = level
        self.fmt = fmt

        self.logger = logging.getLogger(self.filename)
        self.logger.propagate = False  # 防止终端重复打印


        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(self.level))

        if not self.logger.handlers:
            # self.logger.handlers.clear()

            # 设置日志格式
            format_str = logging.Formatter(self.fmt)

            # 设置屏幕打印
            sh = logging.StreamHandler()  # 往屏幕上输出

            # 设置打印格式
            sh.setFormatter(format_str)  # 设置屏幕上显示的格式

            # 把对象加到logger里
            self.logger.addHandler(sh)

            # 往文件里写入handler
            # th = handlers.TimedRotatingFileHandler(filename=filename, backupCount=backCount, encoding='utf-8')
            th = logging.FileHandler(filename, 'a', encoding='utf-8')
            # 设置文件里写入的格式
            th.setFormatter(format_str)

            # 添加写入文件的操作
            self.logger.addHandler(th)

class ImageNetCustom(data.Dataset):  # 新建一个数据集类，并且需要继承PyTorch中的data.Dataset父类
    # 默认构造函数，传入数据集类别（训练或测试），以及数据集路径
    def __init__(self, mode, dir, dataTransform=dataTransform):
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

log = Logger(os.path.join(os.path.join(args.arch, args.work_dir),
                          'torchvision_imagenet_%s.log' % args.arch), level='info')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # if args.gpu is not None:
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # ngpus_per_node = 6
    # print("has gpu nums: ", ngpus_per_node)
    # args.world_size = ngpus_per_node * args.world_size
    # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # 注释全部的gpu资源
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        # torch.cuda.set_device(args.gpu)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        # model = models.__dict__[args.arch].MobileNetV2()

    # if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    #     print('using CPU, this will be slow')
    # elif args.distributed:
    #     # For multiprocessing distributed, DistributedDataParallel constructor
    #     # should always set the single device scope, otherwise,
    #     # DistributedDataParallel will use all available devices.
    #     if torch.cuda.is_available():
    #         if args.gpu is not None:
    #             print("args.distributed: true", args.gpu)
    #             torch.cuda.set_device(args.gpu)
    #             model.cuda(args.gpu)
    #             # When using a single GPU per process and per
    #             # DistributedDataParallel, we need to divide the batch size
    #             # ourselves based on the total number of GPUs of the current node.
    #             args.batch_size = int(args.batch_size / ngpus_per_node)
    #             args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #         else:
    #             model.cuda()
    #
    #             # DistributedDataParallel will divide and allocate batch_size to all
    #             # available GPUs if device_ids are not set
    #             model = torch.nn.parallel.DistributedDataParallel(model)
    # elif args.gpu is not None and torch.cuda.is_available():
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)
    #
    # # elif torch.backends.mps.is_available():
    # #     device = torch.device("mps")
    # #     model = model.to(device)
    # else:
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #         model.features = torch.nn.DataParallel(model.features)
    #         model.cuda()
    #     else:
    #         model = torch.nn.DataParallel(model).cuda()
    #
    # if torch.cuda.is_available():
    #     if args.gpu:
    #         device = torch.device('cuda:{}'.format(args.gpu))
    #     else:
    #         device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler

    # criterion = nn.CrossEntropyLoss().to(device)
    torch.cuda.set_device(args.gpu)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    model, optimizer = amp.initialize(model, optimizer)
    model = DistributedDataParallel(model)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # scheduler = StepLR(optimizer, step_size=2, gamma=0.6)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                # loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(args.gpu)
                print("best acc is:", best_acc1)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        # official数据集使用
        # traindir = os.path.join(args.data, 'train')
        # valdir = os.path.join(args.data, 'val')
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        #
        # train_dataset = datasets.ImageFolder(
        #     traindir,
        #     transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))
        #
        # val_dataset = datasets.ImageFolder(
        #     valdir,
        #     transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

        # 使用自己下载的imagenet数据集

        IMAGE_SIZE = 224
        writer = SummaryWriter(args.arch + 'summaryWriter')
        dataTransform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 尺寸变化
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 中心裁剪
            transforms.ToTensor()  # 归一化
        ])

        train_data = ImageNetCustom("train", args.path, dataTransform)
        train_size = int(0.8 * len(train_data))
        test_size = len(train_data) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    training_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        epoch_time = time.time()
        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        if dist.get_rank() == 0:
            log.logger.info("epoch_time cost: %s" % str(time.time() - epoch_time))

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)


        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict()
            }, is_best, args)
    if dist.get_rank() == 0:
        log.logger.info("train cost time is: %s" % (str(time.time() - training_time)))

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Batch_Time', ':6.3f')
    data_time = AverageMeter('Data_processor', ':6.3f')
    losses = AverageMeter('Loss', ':.6f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    top5 = AverageMeter('Acc@5', ':3.2f')
    lr = AverageMeter("lr", ":.7f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, lr],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    global totalCount
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        # images = images.to(device, non_blocking=True)
        images = images.cuda()
        target = target.cuda()
        # target = target.to(device, non_blocking=True)

        # compute
        output = model(images)
        totalCount += 1
        target = target.squeeze(1)
        loss = criterion(output, target)
        writer.add_scalar("loss", loss, totalCount)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # log.logger.info("after val the acc: {}-{}".format(acc1, acc5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        # lr_ = scheduler.get_last_lr()[0]
        # lr.update(lr_, 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and dist.get_rank() == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        total = 0
        correct = 0
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                target = target.squeeze(1)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                _, pred = torch.max(output.data, 1)
                total += images.shape[0]
                correct += pred.data.eq(target.data).cpu().sum()
                # log.logger.info("top1 is: {}, acc 1 is: {}, acc5 is: {}".format((1.0 * correct.numpy() / total),
                #                                                                 acc1, acc5))

                losses.update(loss.item(), images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0 and dist.get_rank() == 0:
                    progress.display(i + 1)


    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.5', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    # if dist.get_rank() == 0:
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))
    if dist.get_rank() == 0:
        progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, args):
    filename = os.path.join(os.path.join(args.arch, args.work_dir), args.arch+".pth")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        os.path.join(os.path.join(args.arch, args.work_dir), 'model_best.pth'))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'     # 这个打印太啰嗦了
        fmtstr = '{name}: {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        # log.logger.info('\t'.join(entries))
        log.logger.info(' '.join(entries))

    def display_summary(self):
        entries = ["val: *"]
        entries += [meter.summary() for meter in self.meters]
        log.logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
