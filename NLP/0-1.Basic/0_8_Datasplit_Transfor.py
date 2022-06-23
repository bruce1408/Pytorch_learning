import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class MyDataset(data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


init_dataset = data.TensorDataset(
    torch.randn(100, 3, 24, 24),
    torch.randint(0, 10, (100,))
)

lengths = [int(len(init_dataset) * 0.8), int(len(init_dataset) * 0.2)]
subsetA, subsetB = data.dataset.random_split(init_dataset, lengths)
datasetA = MyDataset(subsetA, transform=transforms.Normalize((0., 0., 0.), (0.5, 0.5, 0.5)))
datasetB = MyDataset(subsetB, transform=transforms.Normalize((0., 0., 0.), (0.5, 0.5, 0.5)))
