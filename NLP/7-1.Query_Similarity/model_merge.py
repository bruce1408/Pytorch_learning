import os
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import config.config as cfg
from importlib import import_module
from models import DSSM, LSTMBasic, LSTMBidAtten
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from CustomData.dataset import cut_sentence, read_vocab, CustomData, collate_fn_test
from utils.json_extra import write_json, read_json


def evaluate(model, data_iter, device, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([],   dtype=int)
    labels_all = np.array([], dtype=int)
    cross_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in data_iter:
            first_txt, second_txt, lengths_first, lengths_second, _ = [x.to(device) for x in batch]
            outputs = model(first_txt, second_txt, lengths_first, lengths_second)

            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total


if __name__ == "__main__":
    models = ["BilstmAttn_epoch_1acc_0.640625loss_44.29476088285446",
              "CNN_epoch_3acc_0.644375loss_47.42921340465546",
              "BilstmAttn_epoch_1acc_0.643125loss_44.53440725803375",
              "mulBilstm_epoch_1acc_0.645loss_44.17911180853844",
              "BilstmAttn_epoch_1acc_0.643125loss_44.53440725803375"]

    test_path = "data/outputs/KUAKE-QQR_test.json"
    test_sentences = cut_sentence(test_path, train=False)

    print(test_sentences.__len__())
    vocab = read_vocab("./data/vocab")

    test_data = [(vocab.convert_tokens_to_ids(pairdata[0]), vocab.convert_tokens_to_ids(pairdata[1])) for
     pairdata in test_sentences]

    test_dataset = CustomData(test_data)
    test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_test, shuffle=False)

    # 预测使用的网络

    # 纯CNN
    # model = DSSM(len(vocab), 3)

    # 双向多层LSTM
    # model = LSTMModel(len(vocab), 3)

    # LSTM加Attention
    # model = LSTMAttn(len(vocab), 3)
    totallabel = []
    for modelpath in models:
        model_name = modelpath.split("-")[0]
        print(modelpath)
        if model_name =="mulBilstm":
            net = import_module("models." + "LSTMs")
            model = net.LSTMs.LSTMModel(len(vocab), 3)

            pass
        elif model_name == "BilstmAttn":
            net = import_module("models." + "LSTMAtten")
            model = net.LSTMAttn(len(vocab), 3)
        else:
            # 默认使用CNN
            net = import_module("models." + "CNNs")
            model = net.DSSM(len(vocab), 3)

        model.load_state_dict(torch.load("./checkpoints/"+modelpath))
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for batch in tqdm(test_data_loader):
                first_txt, second_txt, lengths_first, lengths_second = [x.to(device) for x in batch]
                outputs = model(first_txt, second_txt, lengths_first, lengths_second)
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                predict_all = np.append(predict_all, predic)

        print(predict_all.flatten())
        label = predict_all.flatten()
        print(len(label))
        totallabel.append(label)

    # write_json(label)
    print(totallabel)
