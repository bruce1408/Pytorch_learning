import os
from importlib import import_module
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import argparse
import time
from sklearn import metrics
import torch
from models.CNN import DSSM
import torch.nn as nn
from config import config as cfg
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from tqdm.auto import tqdm
from CustomData.dataset import generate_data, generate_vocab, cut_sentence, CustomData, collate_fn
import torch.nn.functional as F

# from utils.network import adjust_learning_rate
# from tensorboardX import SummaryWriter
# from CustomData.customdata import RoiDataset
# from CustomData.customdata import detection_collate


def evaluate(model, data_iter, device, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([],   dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            first_txt, second_txt, lengths_first, lengths_second, labels = [x.to(device) for x in batch]
            outputs = model(first_txt, second_txt, lengths_first, lengths_second)
            loss = cross_loss(outputs, labels)

        # for texts, labels in data_iter:
        #     outputs = model(texts)
        #     loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total


if __name__ == "__main__":
    # 生成单词词典
    path_train = "./data/KUAKE-QQR_train.json"
    path_test = "./data/KUAKE-QQR_dev.json"

    train_data = cut_sentence(path_train)
    val_data = cut_sentence(path_test)

    # 生成词典的过程
    total_sentence = [word for each_pair in train_data for word in each_pair[0]]
    total_sentence += [word for each_pair in train_data for word in each_pair[1]]
    total_sentence += [word for each_pair in val_data for word in each_pair[0]]
    total_sentence += [word for each_pair in val_data for word in each_pair[1]]
    vocab = generate_vocab(total_sentence)

    # # print the len of the vocab
    print(len(vocab))

    train_data, val_data = generate_data(vocab, train_data, val_data)
    # print(train_data.__len__(), val_data.__len__())

    train_dataset = CustomData(train_data)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=True)

    val_dataset = CustomData(val_data)
    val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DSSM(len(vocab), 3)
    model.to(device)
    cross_loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)  # 使用Adam优化器
    model.train()

    total_batch = 0
    for epoch in range(1, cfg.max_epochs):
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            first_txt, second_txt, lengths_first, lengths_second, labels = [x.to(device) for x in batch]
            # print(first_txt.shape, second_txt.shape, labels.shape, lengths.shape)
            outputs = model(first_txt, second_txt, lengths_first, lengths_second)
            loss = cross_loss(outputs, labels)
            # model.zero_grad() # 这种好像也可以
            loss.backward()
            optimizer.step()
            if total_batch % cfg.display_interval == 0:
                # 每多少轮输出在训练集和验证集上的效果
                # msg = 'Iter: {0:>6},  Train Loss: {1:>5.2}'
                # print(msg.format(total_batch, loss.item()))
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, val_data_loader, device)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {' \
                      '4:>6.2%} '
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc))
                # print(dev_acc)
            total_batch += 1

