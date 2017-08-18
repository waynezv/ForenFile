from __future__ import print_function
import argparse
import os
import time
import pdb
import math
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import itertools

from dataloader import dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='timit | tidigits | to be added')
parser.add_argument('--dataroot', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

cudnn.benchmark = True

EPS = 1e-12

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

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.conv1 = nn.ConvTranspose2d(4, 4, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.ConvTranspose2d(4, 4, 3, 1, 1)
        self.relu2 = nn.LeakyReLU(0.2)
        self.drop1 = nn.AlphaDropout(0.5)
        self.conv3 = nn.ConvTranspose2d(4, 2, 3, 2, 0)
        self.relu3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.ConvTranspose2d(2, 1, 3, 2, 0)
        self.relu4 = nn.LeakyReLU(0.2)
        self.pool1 = nn.AdaptiveMaxPool2d((257, 15))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.drop1(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool1(x)

        return x


class _netE(nn.Module):
    def __init__(self):
        super(_netE, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(2, 4, 3, 1, 1)
        self.relu2 = nn.LeakyReLU(0.2)
        self.drop1 = nn.Dropout2d(0.5)
        self.conv3 = nn.Conv2d(4, 4, 3, (2,1), (0,1))
        self.relu3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(4, 4, 3, (2,1), (0,1))
        self.relu4 = nn.LeakyReLU(0.2)
        self.pool1 = nn.AdaptiveMaxPool2d((64,1))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.drop1(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool1(x)

        return x


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.infer_x = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 8, 3, (2,1), (0,1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, 3, (2,1), (0,1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool2d((16,1))
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(4, 4, (3,1), 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 8, (3,1), (2,1), (0,0)),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),
            nn.Conv2d(8, 16, (3,1), (2,1), (0,0)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool2d((16,1))
        )

        self.infer_xz = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x, z):
        x = self.infer_x(x)
        x = x.view(-1, 256, 1, 1)
        z = self.infer_z(z)
        z = z.view(-1, 256, 1, 1)
        out = self.infer_xz(torch.cat((x,z), 1 ))
        return out.view(-1)


# load data
ctlpath = './phctl_out'
phn_list = ('AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH',
            'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH',
            'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG',
            'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH',
            'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH')
featpath = '../phsegwav_feat'

kwargs = {'target': 'speaker_id',
            # 'min_nframe': , 'max_nframe': ,
          'batch': False, 'batch_size': opt.batchSize, 'test_batch_size': opt.batchSize,
            'shuffle': True}

print('Loading data ...')
end = time.time()
dtrain = dataloader(ctlpath, phn_list, featpath, **kwargs)
print('Done. Time: {:.3f}'.format(time.time()-end))

# Define model
netD = _netD()
netG = _netG()
netE = _netE()
netD.apply(weights_init)
netG.apply(weights_init)
netE.apply(weights_init)

criterion = nn.BCELoss()

fixed_z = torch.FloatTensor(1, 4, 64, 1).normal_(0, 1)
label = torch.FloatTensor(1)
real_label = 1
fake_label = 0

if opt.cuda:
    netD = nn.DataParallel(netD, device_ids=[0,1,2,3]).cuda()
    netG = nn.DataParallel(netG, device_ids=[0,1,2,3]).cuda()
    netE = nn.DataParallel(netE, device_ids=[0,1,2,3]).cuda()
    criterion.cuda()
    fixed_z = fixed_z.cuda()

fixed_z = Variable(fixed_z)

# setup optimizer
optimizer_D = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_GE = optim.Adam(itertools.chain(netG.parameters(), netE.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))


for epoch in range(opt.niter):
    for itr, data in enumerate(dtrain):
        x = data['x']
        netD.zero_grad()
        netG.zero_grad()
        netE.zero_grad()

        z = Variable(torch.randn(1, 4, 64, 1).type(torch.cuda.FloatTensor))
        Gz = netG(z)

        real_cpu = torch.from_numpy(x).float()
        x = torch.from_numpy(x).float()
        if opt.cuda:
            x = x.cuda()
        x = Variable(x)
        Ex = netE(x)

        output_g = netD(Gz, z)
        output_e = netD(x, Ex)

        # loss & back propagation
        loss_d = -torch.mean(torch.log(output_e+EPS)+torch.log(1-output_g+EPS))
        loss_ge = -torch.mean(torch.log(output_g+EPS)+torch.log(1-output_e+EPS))

        # loss_d.backward(retain_graph=True)
        loss_d.backward(retain_graph=True)
        optimizer_D.step()
        loss_ge.backward()
        optimizer_GE.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_GE: %.4f D(G(z), z): %.4f D(x, E(x)): %.4f'
              % (epoch, opt.niter, itr, len(dtrain)*opt.batchSize,
                 loss_d.data[0], loss_ge.data[0], output_g.data[0], output_e.data[0]))

        if (itr % 100) == 0:
              vutils.save_image(real_cpu,
                                '%s/real_samples.png' % opt.outf,
                                normalize=True)
              fake = netG(fixed_z)
              vutils.save_image(fake.data,
                                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                                normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
