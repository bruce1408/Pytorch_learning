import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的全连接神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = self.fc(x)
        return x


# 创建原始模型实例
model = SimpleModel()

# 定义训练数据集和优化器
train_data = torch.randn(100, 10)
train_labels = torch.randint(0, 5, (100,))
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 前期训练
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = nn.CrossEntropyLoss()(outputs, train_labels)
    loss.backward()
    optimizer.step()

# 量化模型参数
bit_width = 8  # 量化的位宽
scale_factor = 2 ** bit_width - 1  # 量化缩放因子
with torch.no_grad():
    for param in model.parameters():
        param.mul_(scale_factor).round_().div_(scale_factor)

# 量化感知训练
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = nn.CrossEntropyLoss()(outputs, train_labels)
    loss.backward()
    optimizer.step()

# 评估量化模型的准确性
model.eval()
with torch.no_grad():
    test_data = torch.randn(10, 10)
    test_labels = torch.randint(0, 5, (10,))
    outputs = model(test_data)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
    print(f"Quantized Model Accuracy: {accuracy}")
