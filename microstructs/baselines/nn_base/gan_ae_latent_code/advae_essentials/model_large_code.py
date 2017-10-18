#!/usr/bin/env python
# encoding: utf-8

import pdb
import math
import torch
from torch.autograd import Variable, grad
import torch.nn as nn

class _senetE(nn.Module):
    '''
    Encoder for spectrograms from sentence segments.
    '''
    def __init__(self):
        super(_senetE, self).__init__()
        # x 1*414*450
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False), #200
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), #100
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), #50
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), #25
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), #12
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 2048, 4, 2, 1, bias=False), #6
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2048, 2048, 4, 2, 1, bias=False), #3
            nn.BatchNorm2d(2048),
            nn.AdaptiveMaxPool2d((1, 1)), # 2048*1*1
            nn.LeakyReLU(0.2),
            nn.Conv2d(2048, 200, 1, 1, 0, bias=False) # 200*1*1
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class _senetG(nn.Module):
    '''
    Generate spectrogram of sentence segment from latent code.
    '''
    def __init__(self):
        super(_senetG, self).__init__()
        # z 200*1*1
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(200, 2048, 4, 2, 1, bias=False), #2
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(2048, 2048, 4, 2, 1, bias=False), #4
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(2048, 2048, 4, 2, 1, bias=False), #8
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False), #16
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False), #32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), #64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), #128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), #256
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), #512
            nn.BatchNorm2d(1),
            nn.AdaptiveMaxPool2d((414, 450)) # Gz 1*414*450
        )

    def forward(self, x):
        x = self.generator(x)
        return x


class _senetD(nn.Module):
    def __init__(self):
        super(_senetD, self).__init__()
        # x 1*414*450
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False), #200
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False), #100
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), #50
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), #25
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), #12
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 2, 1, bias=False), #6
            nn.BatchNorm2d(1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x.view(-1)


class _codeNetE(nn.Module):
    def __init__(self):
        super(_codeNetE, self).__init__()
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
            nn.LeakyReLU(0.2)
            # nn.Tanh()
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, 1, 0)
        )
        self.infer_y = nn.Sequential(
            nn.Linear(1024, 630)
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.encoder(x)
        z = self.infer_z(x) # 1024*1*1

        x = x.view(-1, 1024)
        y = self.infer_y(x) # 630
        return z, y


class _codeNetG(nn.Module):
    '''
    Decoder / generator from latent code and condition code.
    '''
    def __init__(self):
        super(_codeNetG, self).__init__()
        # z 1024*1*1 c 630
        self.conditioner = nn.Sequential(
            nn.Linear(630, 4000),
            nn.ReLU(),
            nn.Linear(4000, 4000),
            nn.ReLU(),
            nn.Linear(4000, 1024)
            # nn.Tanh()
        )
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
            nn.AdaptiveMaxPool2d((414, 450)) # Gz 1*414*450
            # nn.Tanh()
        )

    def forward(self, x, c):
        c = self.conditioner(c).view(-1, 1024, 1, 1)
        # TODO: gate
        x = x + c
        x = self.generator(x)
        return x


class _codeNetD(nn.Module):
    '''
    Discriminator for latent codes.

    '''
    def __init__(self):
        super(_codeNetD, self).__init__()
        # z 200*1*1
        self.classifier = nn.Sequential(
            nn.Conv2d(200, 2000, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2000, 2000, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2000, 2000, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2000, 1, 1, 1, 0)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x.view(-1)


class _codeNetDy(nn.Module):
    '''
    Discriminator for latent y.

    '''
    def __init__(self):
        super(_codeNetDy, self).__init__()
        # y 630
        self.classifier = nn.Sequential(
            nn.Linear(630, 2000),
            nn.LeakyReLU(0.2),
            nn.Linear(2000, 2000),
            nn.LeakyReLU(0.2),
            nn.Linear(2000, 2000),
            nn.LeakyReLU(0.2),
            nn.Linear(2000, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x.view(-1)


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
            nn.LeakyReLU(0.2)

            # nn.AdaptiveMaxPool2d((414, 100))
            # output 1*414*100
        )

    def forward(self, x, x_real):
        (_, _, H, W) = x_real.size()
        return nn.functional.adaptive_max_pool2d(self.generator(x), (H, W))


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


def weights_init(m):
    '''
    Custom weights initialization.
    '''
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


def calc_gradient_penalty(netD, real_data, fake_data, penalty=10):
    '''
    Calculate gradient penalty for WGAN-GP.
    '''
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=Variable(torch.ones(disc_interpolates.size()).cuda()),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty

    return gradient_penalty
