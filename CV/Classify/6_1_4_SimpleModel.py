import torch
import torch.nn as nn
from torchsummary import summary


class toyModel(nn.Module):
    def __init__(self):
        """
        卷积 还有 padding 后面的 bool 判断，true 选后面，false 选前面
        """
        super(toyModel, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(1, 3)[True],
            stride=2,
            padding=(0, 1)[False],
            bias=False
        )

    def forward(self, x):
        output = self.conv(x)
        return output


if __name__ == "__main__":

    model = toyModel()
    x = torch.rand((2, 3, 224, 224))
    if torch.cuda.is_available():
        summary(model.cuda(), (3, 224, 224))
        x = x.cuda()
        output = model(x).cuda()
    else:
        summary(model, (3, 224, 224))
        output = model(x)
    print(model)
    print(output.shape)
