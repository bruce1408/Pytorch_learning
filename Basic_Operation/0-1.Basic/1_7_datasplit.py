import random
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

# 1/1/19 - Basic sklearn style splitter to randomize train/test Dataset from a single folder.
# SKlearn has this built in, pytorch does not.
# credit for this goes to Chris Fotache, from this article.
# https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-
# inference-on-single-images-99465a1e9bf5
# added in constants to make adjustments easier.

# constants
DATA_DIR = '../../data/dogs_cats/train'
BATCH_SIZE = 64
IMG_RESIZE_TO = 224


def load_split_train_test(DATA_DIR, valid_size=.2):
    train_transforms = transforms.Compose([transforms.Resize(IMG_RESIZE_TO),
                                           transforms.ToTensor(),
                                           ])
    test_transforms = transforms.Compose([transforms.Resize(IMG_RESIZE_TO),
                                          transforms.ToTensor(),
                                          ])

    # TODO: should add flag to make dir if not present?  Will still fail due to no images but one less step
    train_data = datasets.ImageFolder(DATA_DIR,
                                      transform=train_transforms)

    test_data = datasets.ImageFolder(DATA_DIR,
                                     transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=BATCH_SIZE)

    return trainloader, testloader


trainloader, testloader = load_split_train_test(DATA_DIR, 0.2)
print(trainloader.dataset.classes)



