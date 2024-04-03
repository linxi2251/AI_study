import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="../../data_set", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=8)

for x, y in train_loader:
    print(x)
    print(y)
    
