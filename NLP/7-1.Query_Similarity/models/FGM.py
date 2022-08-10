import torch


#
#
# class FGM():
#     """
#     参考:  https://zhuanlan.zhihu.com/p/91269728
#     """
#     def __init__(self, model):
#         self.model = model
#         self.backup = {}
#
#     def attack(self, epsilon=1., emb_name='word_embeddings'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 self.backup[name] = param.data.clone()
#                 norm = torch.norm(param.grad)
#                 if norm != 0:
#                     r_at = epsilon * param.grad / norm
#                     param.data.add_(r_at)
#
#     def restore(self, emb_name='word_embeddings'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}
#


class FGM():

    def __init__(self, model):

        self.model = model

        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):

        # emb_name这个参数要换成你模型中embedding的参数名

        # 例如，self.emb = nn.Embedding(5000, 100)

        for name, param in self.model.named_parameters():

            if param.requires_grad and emb_name in name:

                self.backup[name] = param.data.clone()

                norm = torch.norm(param.grad)  # 默认为2范数

                if norm != 0:
                    r_at = epsilon * param.grad / norm

                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):

        # emb_name这个参数要换成你模型中embedding的参数名

        for name, param in self.model.named_parameters():

            if param.requires_grad and emb_name in name:
                assert name in self.backup

                param.data = self.backup[name]

        self.backup = {}
