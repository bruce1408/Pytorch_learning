import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity

# 定义一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 生成一些简单的数据进行训练
X = np.random.rand(1000, 2).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 1).astype(np.int64)

train_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型并进行训练
model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
def train(model, loader, optimizer, epochs=15):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

train(model, train_loader, optimizer)

# 测试模型精度
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

accuracy_before = test(model, train_loader)
print(f"Accuracy before quantization: {accuracy_before:.2f}")

# 获取量化前的权重
weights_before = []
for param in model.parameters():
    weights_before.append(param.detach().numpy().flatten())
weights_before = np.concatenate(weights_before)

# 对模型进行量化（这里使用简单的8位量化）
model_quantized = SimpleNet()
model_quantized.load_state_dict(model.state_dict())

with torch.no_grad():
    for param in model_quantized.parameters():
        param.copy_(torch.round(param * 255) / 255)  # 简单的8位量化

# 获取量化后的权重
weights_after = []
for param in model_quantized.parameters():
    weights_after.append(param.detach().numpy().flatten())
weights_after = np.concatenate(weights_after)

# 计算量化前后的余弦相似度
cos_sim = cosine_similarity([weights_before], [weights_after])[0][0]
print(f"Cosine similarity after quantization: {cos_sim:.2f}")

# 测试量化后模型的精度
accuracy_after = test(model_quantized, train_loader)
print(f"Accuracy after quantization: {accuracy_after:.2f}")
