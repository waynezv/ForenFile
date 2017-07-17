from __future__ import print_function
import argparse
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable

from dataloader import dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageHeight', type=int, default=257, help='the height of the input image to network')
parser.add_argument('--imageWidth', type=int, default=17, help='the width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
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

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    # elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal(m.weight.data)
        # nn.init.xavier_normal(m.bias.data)

class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.sub = nn.Sequential(
            nn.Linear(nz, 3120),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution. 16 x 65 x 3
            nn.ConvTranspose2d(ngf, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # state size. 16 x 65 x 3
            nn.ConvTranspose2d(16, 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # state size. 8 x 129 x 5
            nn.ConvTranspose2d(8, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # state size. 8 x 129 x 5
            nn.ConvTranspose2d(8, 4, (3,2), 2, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            # state size. 4 x 257 x 8
            nn.ConvTranspose2d(4, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            # state size. 4 x 257 x 8
            nn.ConvTranspose2d(4, nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 257 x 8
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = nn.parallel.data_parallel(self.sub, input, range(self.ngpu))
            input = input.view(-1, 16, 65, 3)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            input = self.sub(input)
            input = input.view(-1, 16, 65, 3)
            output = self.main(input)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 257 x 8
            nn.Conv2d(nc, 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 4 x 257 x 8
            nn.Conv2d(4, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 4 x 257 x 8
            nn.Conv2d(4, 8, (3,2), 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 8 x 129 x 5
            nn.Conv2d(8, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 8 x 129 x 5
            nn.Conv2d(8, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 16 x 65 x 3
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.sub = nn.Sequential(
            nn.Linear(3120, nz),
            nn.ReLU(True),
            nn.Linear(nz, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            input = input.view(-1, 3120)
            output = nn.parallel.data_parallel(self.sub, input, range(self.ngpu))

        else:
            input = self.main(input)
            input = input.view(-1, 3120)
            output = self.sub(input)

        return output.view(-1, 1)


netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

imgH, imgW = opt.imageHeight, opt.imageWidth
input = torch.FloatTensor(opt.batchSize, 1, imgH, imgW)
noise = torch.FloatTensor(opt.batchSize, nz)
fixed_noise = torch.FloatTensor(opt.batchSize, nz).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# load data
ctlpath = './phctl_out'
phn = 'AA'
featpath = '../phsegwav_feat'


kwargs = {'target': 'speaker_id',
            'min_nframe': 5, 'max_nframe': imgW,
            'batch_size': opt.batchSize, 'test_batch_size': opt.batchSize,
            'shuffle': True}

print('Loading data ...')
end = time.time()
dtrain, _ = dataloader(ctlpath, phn, featpath, **kwargs)
print('Done. Time: {:.3f}'.format(time.time()-end))

for epoch in range(opt.niter):
    i = 0
    for data, _ in dtrain:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = torch.from_numpy(data).float()
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dtrain)*opt.batchSize,
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)
    i += 1

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
