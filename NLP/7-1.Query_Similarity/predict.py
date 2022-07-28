import os
import torch
import time
import torch.nn as nn
import numpy as np
from importlib import import_module
from sklearn import metrics
import config.config as cfg
# from models.DSSM import DSSM
# from models.LSTMBidAtten import LSTMAttn
from transformers import BertTokenizer
from models.Bert import Net
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from CustomData.dataset import cut_sentence, read_vocab, CustomData, collate_fn_test, collate_fn_bert_test, BertData
from utils.json_extra import write_json, read_json


def evaluate(model, data_iter, device, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([],   dtype=int)
    labels_all = np.array([], dtype=int)
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


def predict_result(models, device, bertModel=False):

    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    model_merge = False
    if len(models) > 1:
        model_merge = True
        print("multi model has found")
    merge_labels = []
    model = Net
    for modelname in models:
        model_name_prefix = modelname.split("_")[0]
        model_father = import_module("models." + model_name_prefix)
        if model_name_prefix == "Bert":
            test_dataset = BertData(test_sentences)
            test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_bert_test)
            model = model_father.Net(cfg.pretrain_path, label_size=3)
            tokenizer = BertTokenizer.from_pretrained(cfg.pretrain_path)
        elif model_name_prefix != "Bert":
            vocab = read_vocab("./data/vocab")
            test_data = [(vocab.convert_tokens_to_ids(pairdata[0]), vocab.convert_tokens_to_ids(pairdata[1])) for
                         pairdata in test_sentences]

            test_dataset = CustomData(test_data)
            test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_test, shuffle=False)
            model = model_father.Net(len(vocab), 3)

        model.load_state_dict(torch.load("./checkpoints/" + modelname))
        model.to(device)
        model.eval()

        predict_all = np.array([], dtype=int)

        with torch.no_grad():
            for batch in tqdm(test_data_loader):
                if model_name_prefix == "Bert":
                    first_txt, second_txt = batch

                    inp_first = tokenizer.batch_encode_plus(first_txt, padding=True, return_tensors='pt')
                    inp_second = tokenizer.batch_encode_plus(second_txt, padding=True, return_tensors='pt')

                    outputs = model(inp_first["input_ids"].to(device), inp_second["input_ids"].to(device),
                                    inp_first["attention_mask"].to(device), inp_second["attention_mask"].to(device))
                elif model_name_prefix != "Bert":
                    first_txt, second_txt, lengths_first, lengths_second = [x.to(device) for x in batch]
                    outputs = model(first_txt, second_txt, lengths_first, lengths_second)
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                predict_all = np.append(predict_all, predic)

        merge_labels.append(predict_all.flatten())

    if not model_merge:
        labels = merge_labels[0]
        print(labels)
        print("single model to predict result...")
    else:
        labels = np.array(merge_labels).reshape((-1, 1596))
        labels = np.around(np.mean(labels, axis=0))
        print("multi models to predict result...")
    write_json(labels.astype(int), timestr, model_merge)


if __name__ == "__main__":
    test_path = "data/KUAKE-QQR_test.json"
    test_sentences = cut_sentence(test_path, train=False, bertModel=True)
    print(test_sentences.__len__())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型融合预测
    models = [
        # "DSSM_epoch_4acc_0.643125loss_46.79949390888214",
        "LSTMBasic_epoch_1acc_0.643125loss_44.17954781651497",  # 0.6429
        # "LSTMBidAtten_epoch_1acc_0.643125loss_44.53440725803375",
        # "LSTMMultiLayerBidAttn_epoch_1acc_0.64loss_46.53512938320637",
        "LSTMBasic_epoch_1acc_0.638125loss_44.325793623924255",
        # "Bert_epoch_1acc_0.68375loss_41.274032562971115"
    ]

    predict_result(models, device, bertModel=False)
