import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AdamW, BertForSequenceClassification, BertConfig


class ArgModel(nn.Module):
    def __init__(self, pre_train_path, label_size=3):
        super(ArgModel, self).__init__()
        self.first_bert = Net(pre_train_path, label_size)
        self.second_bert = Net(pre_train_path, label_size)

        self.label_size = label_size
        self.fc = nn.Linear(3 * 768, self.label_size)
        self.name = "Bert"

    def forward(self, first_sentence, second_sentence, first_sentence_mask=None, second_sentence_mask=None):

        first_output = self.first_bert(first_sentence, first_sentence_mask)
        # print('first output: ', first_output.shape)
        second_output = self.second_bert(second_sentence, second_sentence_mask)

        concat_results = torch.cat((first_output, second_output, first_output - second_output), dim=1)
        # print(concat_results.size())

        outputs = self.fc(concat_results)
        return outputs


class Net(nn.Module):
    """
    Bert
    """

    def __init__(self, pretrain_path, label_size, dropout=0.2):
        super(Net, self).__init__()
        self.config = BertConfig.from_pretrained(pretrain_path, num_labels=label_size)
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
    # for name, param in model.named_parameters():
    #     print(name)
    input = torch.randint(0, 8, [1, 8])
    input1 = torch.randint(0, 8, [1, 8])

    a1 = torch.tensor([[1, 2, 3, 4, 5]])
    a2 = torch.tensor([[2, 3, 4, 5, 6]])
    mask1 = torch.tensor([[1, 1, 1, 1, 1]])
    mask2 = torch.tensor([[1, 1, 1, 1, 1]])

    models = ArgModel(pre_train_path=pretrain_path, label_size=3)
    outputs = models(a1, a2, mask1, mask2)
    print(outputs.shape)
