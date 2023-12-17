import torch
import torch.nn as nn


# class DSSM(BaseTower):
#     """DSSM双塔模型"""
#     def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
#                  dnn_hidden_units=(300, 300, 128), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-6,
#                  dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
#         super(DSSM, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
#                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
#                                     device=device, gpus=gpus)
#
#         if len(user_dnn_feature_columns) > 0:
#             self.user_dnn = DNN(compute_input_dim(user_dnn_feature_columns), dnn_hidden_units,
#                                 activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
#                                 use_bn=dnn_use_bn, init_std=init_std, device=device)
#             self.user_dnn_embedding = None
#
#         if len(item_dnn_feature_columns) > 0:
#             self.item_dnn = DNN(compute_input_dim(item_dnn_feature_columns), dnn_hidden_units,
#                                 activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
#                                 use_bn=dnn_use_bn, init_std=init_std, device=device)
#             self.item_dnn_embedding = None
#
#         self.gamma = gamma
#         self.l2_reg_embedding = l2_reg_embedding
#         self.seed = seed
#         self.task = task
#         self.device = device
#         self.gpus = gpus
#
#     def forward(self, inputs):
#         if len(self.user_dnn_feature_columns) > 0:
#             user_sparse_embedding_list, user_dense_value_list = \
#                 self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)
#
#             user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
#             self.user_dnn_embedding = self.user_dnn(user_dnn_input)
#
#         if len(self.item_dnn_feature_columns) > 0:
#             item_sparse_embedding_list, item_dense_value_list = \
#                 self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)
#
#             item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
#             self.item_dnn_embedding = self.item_dnn(item_dnn_input)
#
#         if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
#             score = Cosine_Similarity(self.user_dnn_embedding, self.item_dnn_embedding, gamma=self.gamma)
#             output = self.out(score)
#             return output
#
#         elif len(self.user_dnn_feature_columns) > 0:
#             return self.user_dnn_embedding
#
#         elif len(self.item_dnn_feature_columns) > 0:
#             return self.item_dnn_embedding
#
#         else:
#             raise Exception("input Error! user and item feature columns are empty.")
#
#
# def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
#     sparse_feature_columns = list(
#         filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
#
#     dense_feature_columns = list(
#         filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
#
#     varlen_sparse_feature_columns = list(
#         filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
#
#     if not support_dense and len(dense_feature_columns) > 0:
#         raise ValueError(
#             "DenseFeat is not supported in dnn_feature_columns")
#
#     sparse_embedding_list = [embedding_dict[feat.embedding_name](
#         X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
#         feat in sparse_feature_columns]
#
#     varlen_sparse_embedding_list = get_varlen_pooling_list(embedding_dict, X, self.feature_index,
#                                                            varlen_sparse_feature_columns, self.device)
#
#     dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
#                         dense_feature_columns]
#
#     return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list



# class DSSM(nn.Module):
#
#     def __init__(self, dropout=0.2, device="gpu"):
#         super(DSSM, self).__init__()
#         self.device = device
#         self.embed = nn.Embedding(7901, 100)
#         self.fc1 = nn.Linear(100, 256)
#         self.fc2 = nn.Linear(256, 512)
#         self.fc3 = nn.Linear(512,256)
#         self.dropout = nn.Dropout(dropout)
#         self.Sigmoid = nn.Sigmoid() #method1
#         self.relu = nn.ReLU()
#
#     def forward(self, a, b):
#         a = self.embed(a).sum(1)
#         b = self.embed(b).sum(1)
#
#         a = self.relu(self.fc1(a)) #torch.tanh
#         # a = self.dropout(a)
#         a = self.relu(self.fc2(a))
#         # a = self.dropout(a)
#         a = self.relu(self.fc3(a))
#         # a = self.dropout(a)
#
#         b = self.relu(self.fc1(b))
#         # b = self.dropout(b)
#         b = self.relu(self.fc2(b))
#         # b = self.dropout(b)
#         b = self.relu(self.fc3(b))
#         # b = self.dropout(b)
#
#         cosine = torch.cosine_similarity(a, b, dim=1, eps=1e-8)  #计算两个句子的余弦相似度
#         # cosine = self.Sigmoid(cosine-0.5)
#         cosine = self.relu(cosine)
#         cosine = torch.clamp(cosine,0,1)
#         return cosine



class DSSM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, dropout):
        super(DSSM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        a = self.embed(a).sum(1)
        b = self.embed(b).sum(1)

        a = self.dropout(torch.tanh(self.fc1(a)))
        a = self.dropout(torch.tanh(self.fc2(a)))
        a = self.dropout(torch.tanh(self.fc3(a)))

        b = self.dropout(torch.tanh(self.fc1(b)))
        b = self.dropout(torch.tanh(self.fc2(b)))
        b = self.dropout(torch.tanh(self.fc3(b)))

        cosine = torch.cosine_similarity(a, b, dim=1, eps=1e-8)  # 计算两个句子的余弦相似度
        return cosine

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    a = torch.randint(0, 3, (2, 3))
    b = torch.randint(0, 3, (2, 3))
    model = DSSM(30, 100, 0.2)
    output = model(a, b)
    model._init_weights()
    print(model)
    print('output is ',  output)

