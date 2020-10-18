"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
"""
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

import config
from data import MNISTM, BSDS500
from models import Net
from utils import GrayscaleToRgb, GradientReversal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    model = Net().to(device)
    # model.load_state_dict(torch.load(args.MODEL_FILE))
    feature_extractor = model.feature_extractor
    clf = model.classifier

    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 1
    source_dataset = MNIST("/home/bruce/PycharmProjects/Open_set_domain_adaptation/dataset/data",
                           train=True, download=False,
                           transform=Compose(
                               [GrayscaleToRgb(),
                                ToTensor()]))
    source_loader = DataLoader(source_dataset, batch_size=half_batch, shuffle=True, num_workers=8, pin_memory=True)
    
    target_dataset = MNISTM(train=False)
    target_loader = DataLoader(target_dataset, batch_size=half_batch, shuffle=True, num_workers=8, pin_memory=True)

    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

    for epoch in range(1, args.epochs+1):
        # batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))
        # n_batches = 313
        # print("n batch is: ", n_batches)
        total_domain_loss = total_label_accuracy = 0
        for (source_x, source_labels), (target_x, _) in zip(source_loader, target_loader):
            x = torch.cat([source_x, target_x])  # [batch * 2, 3, 28, 28]
            x = x.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]), torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            label_y = source_labels.to(device)

            features = feature_extractor(x).view(x.shape[0], -1)  # [4, 320]
            domain_preds = discriminator(features).squeeze()  # [4]
            label_preds = clf(features[:source_x.shape[0]])  # [2, 10]

            domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
            label_loss = F.cross_entropy(label_preds, label_y)
            loss = domain_loss + label_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_domain_loss += domain_loss.item()
            total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, 'f'source_accuracy={mean_accuracy:.4f}')

        torch.save(model.state_dict(), 'trained_models/revgrad.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    # arg_parser.add_argument('MODEL_FILE', default="PATH", help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=32)
    arg_parser.add_argument('--epochs', dest='epochs', type=int, default=150)
    args = arg_parser.parse_args()
    main(args)
