#!/usr/bin/env python
# encoding: utf-8

import pdb
import math
import torch
import torch.nn as nn

class _senetG(nn.Module):
    def __init__(self):
        super(_senetG, self).__init__()
        # z 1024*1*1
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1), #2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), #4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), #8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), #16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), #32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), #64
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), #128
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 4, 4, 2, 1), #256
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(4, 1, 4, 2, 1), #512
            nn.BatchNorm2d(1),
            nn.AdaptiveMaxPool2d((414, 450)), # Gz 1*414*450
            nn.Tanh()
        )

    def forward(self, x):
        x = self.generator(x)
        return x


class _senetE(nn.Module):
    def __init__(self):
        super(_senetE, self).__init__()
        # x 1*414*450
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), #200
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), #100
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), #50
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), #25
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), #12
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1), #6
            nn.BatchNorm2d(1024),
            nn.AdaptiveMaxPool2d((1, 1)), # Ex 1024*1*1
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class _senetD(nn.Module):
    def __init__(self):
        super(_senetD, self).__init__()
        # x 1*414*450  z 1024*1*1
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), #200
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), #100
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), #50
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), #25
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), #12
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 2, 1), #6
            nn.BatchNorm2d(1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x.view(-1)


class _phnetG(nn.Module):
    def __init__(self):
        super(_phnetG, self).__init__()
        # input 64*16*4
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3,1), 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, (1,3), 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, 2, 0),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 32, (3,1), 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, (1,3), 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 3, 2, 0),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 16, (3,1), 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 16, (1,3), 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, 3, 2, 0),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(8, 8, (3,1), 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 8, (1,3), 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 4, 3, 2, 0),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(4, 4, (3,1), 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(4, 4, (1,3), 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(4, 1, 3, 2, 0),
            nn.LeakyReLU(0.2),

            # nn.AdaptiveMaxPool2d((414, 100))
            # output 1*414*100
        )

    def forward(self, x, x_real):
        (_, _, H, W) = x_real.size()
        return nn.functional.adaptive_max_pool2d(self.generator(x), (H, W))


class _phnetE(nn.Module):
    def __init__(self):
        super(_phnetE, self).__init__()
        # input 1*414*(4-38)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, (3,1), 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, (1,3), 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, (3,1), 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, (1,3), 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.AdaptiveMaxPool2d((16,4))
            # output 64*16*4
        )

    def forward(self, x):
        return self.encoder(x)


class _phnetD(nn.Module):
    def __init__(self):
        super(_phnetD, self).__init__()
        # x 1*414*(4-38)
        self.infer_x = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool2d((16,1))
            # 16*16*1
        )

        # z 64*16*4
        self.infer_z = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool2d((16,1))
            # 16*16*1
        )

        self.infer_xz = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, z):
        x = self.infer_x(x)
        x = x.view(-1, 256)
        z = self.infer_z(z)
        z = z.view(-1, 256)
        out = self.infer_xz(torch.cat((x,z), 1))
        return out.view(-1)


# custom weights initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()
        # nn.init.xavier_normal(m.weight.data)
        # nn.init.xavier_normal(m.bias.data)
