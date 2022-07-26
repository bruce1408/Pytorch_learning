import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AdamW, BertForSequenceClassification, BertConfig


class Net(nn.Module):
    """
    Bert
    """
    def __init__(self, pretrain_path, label_size, dropout=0.2):
        super(Net, self).__init__()
        self.config = BertConfig.from_pretrained(pretrain_path, num_labels=3)
        self.bert = BertModel.from_pretrained(pretrain_path, config=self.config)
        self.name = "Bert"

    def forward(self, inp, attention_mask=None):
        if attention_mask is not None:
            bert_outputs = self.bert(inp, attention_mask)
            output_tokens_embeddings = bert_outputs[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(output_tokens_embeddings.size()).float()
            sum_embeddings = torch.sum(output_tokens_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            vector = sum_embeddings / sum_mask
            return vector
        else:
            _, pooled = self.bert(inp)
            return pooled


if __name__ == "__main__":

    pretrain_path = "/Users/bruce/PycharmProjects/cail2020/pretrain_model/ERNIE_1.0_max-len-512-pytorch"
    model = Net(pretrain_path, 3)
    for name, param in model.named_parameters():
        print(name)