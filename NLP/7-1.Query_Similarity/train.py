import os
from importlib import import_module
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import numpy as np
import random, time
from sklearn import metrics
import torch
import argparse
from models.label_smooth import LabelSmoothingLoss
from models.FGM import FGM
from models.DSSM import Net
from models.LSTMBasic import Net
from models.LSTMBid import Net
from models.LSTMBidAtten import Net
# from models.LSTMMultiLayerBidAttn import Net
# from models.Bert import Net
import torch.nn as nn
from config import config as cfg
from torch.utils.data import DataLoader
from torch import optim
from transformers import BertTokenizer
from tqdm.auto import tqdm
from CustomData.dataset import generate_data, BertData, cut_sentence, CustomData, collate_fn, read_vocab
from CustomData.dataset import collate_fn_bert


def randomSeed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def evaluate(model, data_iter, device, bertModel=False, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([],   dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            if bertModel:
                first_txt, second_txt, labels = batch
                labels = labels.to(device)
                inp_first = tokenizer.batch_encode_plus(first_txt, padding=True, return_tensors='pt')
                inp_second = tokenizer.batch_encode_plus(second_txt, padding=True, return_tensors='pt')

                outputs = model(inp_first["input_ids"].to(device), inp_second["input_ids"].to(device),
                                inp_first["attention_mask"].to(device), inp_second["attention_mask"].to(device))

                # loss = cross_loss(outputs, labels)

            else:
                first_txt, second_txt, lengths_first, lengths_second, labels = [x.to(device) for x in batch]
                outputs = model(first_txt, second_txt, lengths_first.to('cpu'), lengths_second.to('cpu'))
            # loss = cross_loss(outputs, labels)
            loss = label_smoothing(outputs, labels)

            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total


parser = argparse.ArgumentParser(description='similarity pair')
parser.add_argument('--model', type=str, default='Bert', help='choose a model: "DSSM", "LSTMBasic",'
                                                              ' "LSTMBid", "LSTMBidAtten",'
                                                              ' "LSTMMultiLayerBidAttn", "Bert"')
args = parser.parse_args()


if __name__ == "__main__":
    # 生成单词词典
    path_train = "data/KUAKE-QQR_train.json"
    path_test = "data/KUAKE-QQR_dev.json"
    randomSeed(0)
    # there are some models to choose
    models_names = ["DSSM", "LSTMBasic", "LSTMBid", "LSTMBidAtten", "LSTMMultiLayerBidAttn", "Bert"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_batch = 0
    valid_best_acc = 0.63
    # dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数目
    flag = False  # 记录是否很久没有效果提升
    use_bert = False
    # 目录存在的时候不会创建
    os.makedirs(cfg.save_path, exist_ok=True)

    model_name_prefix = args.model
    model_father = import_module("models." + model_name_prefix)
    if model_name_prefix == "Bert":
        use_bert = True
        train_data = cut_sentence(path_train, bertModel=True)
        val_data = cut_sentence(path_test, bertModel=True)

        train_dataset = BertData(train_data)
        val_dataset = BertData(val_data)

        train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn_bert,
                                       shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn_bert)

        model = model_father.Net(cfg.pretrain_path, label_size=3)

        tokenizer = BertTokenizer.from_pretrained(cfg.pretrain_path)


    else:
        train_data = cut_sentence(path_train)
        val_data = cut_sentence(path_test)
        print("loading the vocab from local file...")
        vocab = read_vocab("./data/vocab")
        # print(vocab.token_to_idx)

        train_data, val_data = generate_data(vocab, train_data, val_data)
        train_dataset = CustomData(train_data)
        train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=True)

        val_dataset = CustomData(val_data)
        val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn)
        model = model_father.Net(len(vocab), 3)

    model.to(device)
    # cross_loss = nn.CrossEntropyLoss()
    label_smoothing = LabelSmoothingLoss(3, 0.01)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)  # 使用Adam优化器
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

    model.train()

    for epoch in range(1, cfg.max_epochs):
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            if model_name_prefix == "Bert":
                first_txt, second_txt, labels = batch
                labels = labels.to(device)

                inp_first = tokenizer.batch_encode_plus(first_txt, padding=True, return_tensors='pt')
                inp_second = tokenizer.batch_encode_plus(second_txt, padding=True, return_tensors='pt')

                outputs = model(inp_first["input_ids"].to(device), inp_second["input_ids"].to(device),
                                inp_first["attention_mask"].to(device), inp_second["attention_mask"].to(device))
            else:

                first_txt, second_txt, lengths_first, lengths_second, labels = [x.to(device) for x in batch]
                outputs = model(first_txt, second_txt, lengths_first.to("cpu"), lengths_second.to("cpu"))

            # loss = cross_loss(outputs, labels)
            loss = label_smoothing(outputs, labels)
            # model.zero_grad() # 这种好像也可以
            loss.backward()
            fgm = FGM(model)
            fgm.attack()
            fgm.restore()

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            if total_batch % cfg.display_interval == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                valid_acc, valid_loss = evaluate(model, val_data_loader, device, use_bert)
                model.train()
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  ' \
                      'Val Acc: {4:>6.2%} '
                print(msg.format(total_batch, loss.item(), train_acc, valid_loss, valid_acc))
                if valid_acc > valid_best_acc:
                    valid_best_acc = valid_acc
                    valid_best_loss = valid_loss

                    # 模型保存命名
                    timestr = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                    save_name = os.path.join(cfg.save_path, model.name + "_" + str(device) + "_" + \
                                             timestr + "_epoch_" + str(epoch) +
                                             "_acc_" + str(valid_acc) + "loss_" + str(valid_loss))
                    # 只保存权重即可
                    torch.save(model.state_dict(), save_name)
                    print("save best model, valid_acc:{}".format(valid_acc))
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""
            total_batch += 1
            if total_batch - last_improve > cfg.require_improvement:
                # 验证集loss超过 1000 batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            if flag:
                break

