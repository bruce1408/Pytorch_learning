from torchvision import transforms

from .mnist import *
from .svhn import *
from .usps import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def get_dataset(task):
    """
    :param task: "s2m', 'u2m', 'm2u'"
    :return:
    """
    if task == 's2m':
        train_dataset = SVHN('../Dataset', split='train', download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

        test_dataset = MNIST('../Dataset', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.Lambda(lambda x: x.convert("RGB")),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
    elif task == 'u2m':
        train_dataset = USPS('/home/bruce/PycharmProjects/openset-DA/Dataset', train=True, download=False,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(28, padding=4),
                                 transforms.RandomRotation(10),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))
                             ]))

        test_dataset = MNIST('../Dataset', train=True, download=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))
                             ]))
    else:  # m2u
        train_dataset = MNIST('../Dataset', train=True, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))
                              ]))

        test_dataset = USPS('/home/bruce/PycharmProjects/openset-DA/Dataset/', train=False, download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ]))

    return relabel_dataset(train_dataset, test_dataset, task)


def relabel_dataset(train_dataset, test_dataset, task):
    # print(train_dataset.train_data)
    image_path = []
    image_label = []
    if task == 's2m':
        for i in range(len(train_dataset.data)):
            if int(train_dataset.labels[i]) < 5:
                image_path.append(train_dataset.data[i])
                image_label.append(train_dataset.labels[i])
        train_dataset.data = image_path
        train_dataset.labels = image_label
    else:
        for i in range(len(train_dataset.data)):
            if int(train_dataset.targets[i]) < 5:
                image_path.append(train_dataset.data[i])
                image_label.append(train_dataset.targets[i])
        train_dataset.data = image_path
        train_dataset.targets = image_label

    for i in range(len(test_dataset.data)):
        if int(test_dataset.targets[i]) >= 5:
            test_dataset.targets[i] = 5

    return train_dataset, test_dataset


# if __name__ == "__main__":
#     source_dataset, target_dataset = get_dataset("m2u")
