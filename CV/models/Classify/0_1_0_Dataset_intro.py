"""
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
"""
import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
print("x: ", x)
y = torch.linspace(10, 1, 10)
print("y: ", y)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=5,
    shuffle=False,
    num_workers=2,
)
print(loader)
loader = iter(loader)
input_var = next(loader)[0]
print("next: ", input_var)
print(next(loader))
input_var = input_var.to(torch.float32)[:1]
print(input_var)
# def show_batch():
#     for epoch in range(3):
#         for step, (batch_x, batch_y) in enumerate(loader):
#             # training
#
#             print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
#
#
# if __name__ == '__main__':
#     show_batch()
