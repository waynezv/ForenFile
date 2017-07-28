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
        self.conv1 = nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveMaxPool2d((64, 2), return_indices=False)
        self.fc1 = nn.Linear(32*64*2, 630)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return F.log_softmax(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3)
        # self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 630)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        flat_dim = self._flat_dim(x)
        x = x.view(-1, flat_dim)

        m = nn.Linear(flat_dim, 2048).cuda()
        x = F.relu(m(x))

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc2(x)
        x = self.fc3(x)

        return F.log_softmax(x)

    def _flat_dim(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        return np.prod(size)
