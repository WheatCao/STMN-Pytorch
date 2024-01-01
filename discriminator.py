import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.mask = torch.Tensor([
        #     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
        ###cifar10
        # self.head1 = nn.Linear(100, 20)
        # self.head2 = nn.Linear(20, 1)
        ###cifar100
        self.head1 = nn.Linear(10000, 100)
        self.head2 = nn.Linear(100, 1)

    def forward(self,x):
        x = torch.flatten(x,1)
        x = self.head1(x)
        x = torch.sigmoid(x)
        x = self.head2(x)
        return x