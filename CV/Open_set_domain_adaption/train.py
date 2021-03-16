from __future__ import print_function
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import warnings
from datasets.get_dataset import get_dataset
import models
import utils
warnings.filterwarnings('ignore')
NUM_CLASSES = 6


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Openset-DA SVHN -> MNIST Example')

    parser.add_argument('--task', choices=['s2m', 'u2m', 'm2u'], default='s2m', help='type of task')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 128)')

    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.001)')

    parser.add_argument('--lr-rampdown-epochs', default=201, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

    parser.add_argument('--grl-rampup-epochs', default=20, type=int, metavar='EPOCHS',
                        help='length of grl rampup')

    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')

    parser.add_argument('--th', type=float, default=0.5, metavar='TH',
                        help='threshold (default: 0.5)')

    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--gpu', default='0', type=str, metavar='GPU',
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    return args


args = parse_args()
print(args.task)
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

source_dataset, target_dataset = get_dataset(args.task)

source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

model = models.Net(task=args.task).cuda()

optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                            nesterov=True)

if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

criterion_bce = nn.BCELoss()
criterion_cel = nn.CrossEntropyLoss()

best_prec1 = 0
best_pred_y = []
best_gt_y = []
global_step = 0
total_steps = args.grl_rampup_epochs * len(source_loader)


def train(epoch):
    model.train()
    global global_step
    for batch_idx, (batch_s, batch_t) in enumerate(zip(source_loader, target_loader)):
        adjust_learning_rate(optimizer, epoch, batch_idx, len(source_loader))
        p = global_step / total_steps
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        data_s, target_s = batch_s
        data_t, target_t = batch_t

        data_s, target_s = data_s.cuda(), target_s.cuda(non_blocking=True)
        data_t, target_t = data_t.cuda(), target_t.cuda(non_blocking=True)

        batch_size_s = len(target_s)
        batch_size_t = len(target_t)

        optimizer.zero_grad()

        output_s = model(data_s)
        output_t = model(data_t, constant=constant, adaption=True)

        loss_cel = criterion_cel(output_s, target_s)

        output_t_prob_unk = F.softmax(output_t, dim=1)[:, -1]
        loss_adv = criterion_bce(output_t_prob_unk, torch.tensor([args.th] * batch_size_t).cuda())

        loss = loss_cel + loss_adv

        loss.backward()
        optimizer.step()

        global_step += 1

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tConstant: {:.4f}'.format(
                epoch, batch_idx * args.batch_size, len(source_loader.dataset),
                       100. * batch_idx / len(source_loader), loss.item(), constant))


def val(epoch):
    global best_prec1
    model.eval()
    loss = 0
    pred_y = []
    true_y = []

    correct = 0
    ema_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(target_loader):
            data, target = data.cuda(), target.cuda(non_blocking=True)
            output = model(data)

            loss += criterion_cel(output, target).item()  # sum up batch loss

            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            for i in range(len(pred)):
                pred_y.append(pred[i].item())
                true_y.append(target[i].item())

            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(target_loader.dataset)

    utils.cal_acc(true_y, pred_y, NUM_CLASSES)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(target_loader.dataset),
        100. * correct / len(target_loader.dataset)))

    prec1 = 100. * correct / len(target_loader.dataset)
    if epoch % 1 == 0:
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        if is_best:
            global best_gt_y
            global best_pred_y
            best_gt_y = true_y
            best_pred_y = pred_y


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    lr *= utils.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


try:
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        val(epoch)
    print("------Best Result-------")
    utils.cal_acc(best_gt_y, best_pred_y, NUM_CLASSES)
except KeyboardInterrupt:
    print("------Best Result-------")
    utils.cal_acc(best_gt_y, best_pred_y, NUM_CLASSES)
