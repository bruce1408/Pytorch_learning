import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from code.dataset import get_dataset
import sys


def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    # liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print(
                    "\rIteration: {}/{}, Loss: {}.".format(
                        i + 1, len(dataloaders[phase]), loss.item() * inputs.size(0)
                    ),
                    end="",
                )
                sys.stdout.flush()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "train":
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print("Train Loss: {:.4f} Acc: {:.4f}".format(avg_loss, t_acc))
        print("Val Loss: {:.4f} Acc: {:.4f}".format(val_loss, val_acc))
        print("Best Val Accuracy: {}".format(best_acc))
        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# model = models.resnet18(pretrained=True)
model = models.mobilenet_v2(pretrained=True)
# Finetune Final few layers to adjust for tiny imagenet input
# model.avgpool = nn.AdaptiveAvgPool2d(1)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 200)

model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 200)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Multi GPU
model = torch.nn.DataParallel(model, device_ids=[0, 7])

# Loss Function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

train_dataset, val_dataset, _ = get_dataset()

train_loaders = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=8
)
val_loaders = torch.utils.data.DataLoader(
    val_dataset, batch_size=128, shuffle=True, num_workers=8
)


dataloaders = {}
dataloaders["train"] = train_loaders
dataloaders["val"] = val_loaders

dataset_sizes = {}
dataset_sizes["train"] = len(train_dataset)
dataset_sizes["val"] = len(val_dataset)

model = train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    num_epochs=15,
)

torch.save(model, "models/mobilev2_model.pth")
