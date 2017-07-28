#!/usr/bin/env python
# encoding: utf-8

import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Spatial_MLP(nn.Module):
    def __init__(self):
        super(Spatial_MLP, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.pool = nn.AdaptiveMaxPool2d(16, 16)
        self.fc1 = nn.Linear(32*16*16, 1024)
        self.fc2 = nn.Linear(1024, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x[0] # get rid of indices ??
        # pdb.set_trace()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
