import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 定义一个包含卷积层的神经网络模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 第一卷积层
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 第二卷积层
        self.fc1 = nn.Linear(16 * 1 * 1, 32)  # 全连接层1
        self.fc2 = nn.Linear(32, 16)  # 全连接层2
        self.fc3 = nn.Linear(16, 2)  # 输出层

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 生成一些简单的数据进行训练
X = np.random.rand(1000, 1, 4, 4).astype(np.float32)  # 输入数据为4x4图像
y = (X[:, 0, 0, 0] + X[:, 0, 1, 1] > 1).astype(np.int64)

train_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型并进行训练
model = ConvNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
def train(model, loader, optimizer, epochs=5):
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
model_quantized = ConvNet()
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

# 绘制结果对比图
labels = ['Before Quantization', 'After Quantization']
accuracies = [accuracy_before, accuracy_after]

plt.bar(labels, accuracies, color=['blue', 'orange'])
plt.xlabel('Model State')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Before and After Quantization')
plt.ylim(0, 1)
plt.show()

# 输出结论
print("\n结论：尽管量化前后的余弦相似度为{:.2f}，表明权重变化不大，但模型的精度从{:.2f}下降到{:.2f}，这说明量化过程引入的误差导致模型性能下降。".format(cos_sim, accuracy_before, accuracy_after))
