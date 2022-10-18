import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from model import pyramidnet
import argparse
# from tensorboardX import SummaryWriter
from CV.utils.logger import Logger
parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='断点训练')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0, 1], help="gpu设备编号")

parser.add_argument('--gpu', default=2, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='GPU通信方式用nccl')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='总共的进程数目')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument("--output", default="./dist_output", help="the path of save model")
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices


def main():
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    ngpus_per_node = torch.cuda.device_count()
    print("has gpu nums: ", ngpus_per_node)
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    print("main worker: ", gpu)
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(args.gpu))
    print(args.gpu, args.rank)

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print('==> Making model..')
    net = pyramidnet()
    torch.cuda.set_device(args.gpu)
    net.cuda(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print('The number of parameters of model is', num_params)

    print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # dataset_train = []
    # if dist.get_rank() == 0:
    #     print(0)
    dataset_train = CIFAR10(root='/home/cuidongdong', train=True, download=False, transform=transforms_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.num_workers,
                              sampler=train_sampler)

    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)

    if len(os.listdir(args.output)) != 0:
        print("pretrained model has exist!")
        checkpoints = torch.load(os.path.join(args.output, "last.pth"), map_location="cpu")
        state_dict = checkpoints['model']
        net.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoints["optimizer"])
    train(net, criterion, optimizer, train_loader, args.gpu)


def train(net, criterion, optimizer, train_loader, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100 * correct / total

        batch_time = time.time() - start

        # 不会重复打印
        if batch_idx % 20 == 0 and dist.get_rank() == 0:
            print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                batch_idx, len(train_loader), train_loss / (batch_idx + 1), acc, batch_time))
        if batch_idx % 100 == 0 and dist.get_rank() == 0:
            state = {
                "model": net.state_dict(),
                "batch": batch_idx,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, os.path.join(args.output, "last.pth"))
            # print("model has been saved!")

    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))


if __name__ == '__main__':
    total_start_time = time.time()
    main()
    total_end_time = time.time()
    print("cost time :", total_end_time - total_start_time)