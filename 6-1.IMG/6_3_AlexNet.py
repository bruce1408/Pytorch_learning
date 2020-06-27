import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


if __name__ == '__main__':
    net = Net()
    summary(net, (3, 224, 224))
    # input = torch.randn(6, 3, 224, 224)
    # nn.Conv2d(3, 96, 11, 4),  # 54 x 54
    # nn.ReLU(),
    # nn.MaxPool2d(3, 2),  # 26 x 26
    # nn.Conv2d(96, 256, 5, 1, 2),  # 26 x  26
    # nn.ReLU(),
    # nn.MaxPool2d(3, 2),  # 12 x 12
    # nn.Conv2d(256, 384, 3, 1, 1),  # 12 x 12
    # nn.ReLU(),
    # nn.Conv2d(384, 384, 3, 1, 1),  # 12  x 12
    # nn.ReLU(),
    # nn.Conv2d(384, 256, 3, 1, 1),  # 12 x 12
    # nn.ReLU(),
    # nn.MaxPool2d(3, 2)  # 5 x 5

    # output = conv(input)
    # print(output.shape)