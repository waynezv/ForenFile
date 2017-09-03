#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import time
import pdb
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
from tensorboard_logger import configure, log_value

from args import parser
from dataloader import multithread_loader
from model import _senetE, _senetG, _senetD, weights_init

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

# load data
ctlpath = './phctl_out'
sen_ctl = 'all_sentence_constq.ctl'
featpath = '/mnt/data/wenboz/sentence_constq_feats'

print('Loading data ...')
train_loader = multithread_loader(sen_ctl, featpath, batch_size=opt.batchSize)
SPK_ID_INTEREST = 5

# Init model
netD = _senetD()
netG = _senetG()
netE = _senetE()
netD.apply(weights_init)
netG.apply(weights_init)
netE.apply(weights_init)
print(netG)
print(netE)
print(netD)

fixed_z = torch.FloatTensor(opt.batchSize, 1024, 1, 1).normal_(0, 1)
# label = torch.FloatTensor(1)
# real_label = 1
# fake_label = 0

if opt.cuda:
    netD = nn.DataParallel(netD, device_ids=[0]).cuda()
    netG = nn.DataParallel(netG, device_ids=[0]).cuda()
    netE = nn.DataParallel(netE, device_ids=[0]).cuda()
    fixed_z = fixed_z.cuda()

fixed_z = Variable(fixed_z)

# setup optimizer
optimizer_D = optim.Adam(netD.parameters(), lr=0.1, betas=(opt.beta1, 0.999))
optimizer_G = optim.Adam(netG.parameters(), lr=0.1, betas=(opt.beta1, 0.999))
optimizer_E = optim.Adam(netE.parameters(), lr=0.1, betas=(opt.beta1, 0.999))

optimizer_DG = optim.Adam(itertools.chain(netD.parameters(), netG.parameters()), lr=0.01, betas=(opt.beta1, 0.999))
optimizer_GE = optim.Adam(itertools.chain(netG.parameters(), netE.parameters()), lr=0.01, betas=(opt.beta1, 0.999))

configure(opt.outf)
updt_cnt = 0
for epoch in range(opt.niter):
    for itr, (x, y) in enumerate(train_loader):
        netD.zero_grad()
        netG.zero_grad()
        netE.zero_grad()

        real_cpu = x
        x_real = x
        if opt.cuda:
            x_real = x_real.cuda()
        x_real = Variable(x_real)

        # z_s -> x_fake
        z_s = Variable(torch.randn(x_real.size(0), 1024, 1, 1).type(torch.cuda.FloatTensor))
        x_fake = netG(z_s)

        # D(x)
        output_fake = netD(x_fake)
        output_real = netD(x_real)

        # x_real -> z_r -> x_reconstruct
        z_r = netE(x_real)
        x_reconstruct = netG(z_r)

        # loss & back propagation
        x_int_mask = torch.FloatTensor(x_real.size()).zero_()
        int_indices = y.eq(SPK_ID_INTEREST).nonzero().numpy()
        if int_indices.size != 0:
            x_int_mask[int_indices, :, :, :] = 1
        x_int_mask = Variable(x_int_mask.cuda())

        x_nonint_mask = torch.FloatTensor(x_real.size()).zero_()
        nonint_indices = y.ne(SPK_ID_INTEREST).nonzero().numpy()
        if nonint_indices.size != 0:
            x_nonint_mask[nonint_indices, :, :, :] = 1
        x_nonint_mask = Variable(x_nonint_mask.cuda())

        interest_mask = Variable(y.eq(SPK_ID_INTEREST)).type_as(output_real)
        non_interest_mask = Variable(y.ne(SPK_ID_INTEREST)).type_as(output_real)

        # loss for generator
        loss_g1 = ( x_fake - x_real*x_int_mask ).pow(2).mean()
        loss_g2 = -( x_fake - x_real*x_nonint_mask ).pow(2).mean()
        loss_g3 = -0.5*( output_fake + EPS ).log().mean() - 0.5*( 1 - output_real*interest_mask + EPS ).log().mean()
        # print("loss_g1: {:.4f}".format(loss_g1.data[0]))
        # print("loss_g2: {:.4f}".format(loss_g2.data[0]))
        # print("loss_g3: {:.4f}".format(loss_g3.data[0]))
        loss_g = loss_g3

        # loss for discriminator
        # loss_d = -0.5*( output_real*interest_mask + EPS ).log().mean() - 0.5*( 1 - output_fake + EPS ).log().mean()
        loss_d = -0.5*( output_real + EPS ).log().mean() - 0.5*( 1 - output_fake + EPS ).log().mean()

        # loss for encoder
        loss_e = (x_reconstruct - x_real).pow(2).mean() + z_r.pow(2).mean()

        # TODO: update scheme
        loss_d.backward(retain_graph=True)
        optimizer_DG.step()

        loss_e.backward()
        optimizer_GE.step()

        updt_cnt = updt_cnt + 1
        if updt_cnt == 10:
            updt_cnt = 1

        # TODO: average loss over 100 itrs
        print('[{:d}/{:d}][{:d}/{:d}] Loss_D: {:.4f} Loss_G: {:.4f} Loss_E: {:.4f}'.format(
            epoch, opt.niter, itr, len(train_loader)*opt.batchSize,
            loss_d.data[0], loss_g.data[0], loss_e.data[0]))

        if (itr % 100) == 0:
              vutils.save_image(real_cpu,
                                '%s/real_samples.png' % opt.outf,
                                normalize=False)
              fake = netG(fixed_z)
              vutils.save_image(fake.data,
                                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                                normalize=False)
              x_reconstruct = netG(netE(x_real))
              vutils.save_image(x_reconstruct.data,
                                '%s/real_reconstruct_epoch_%03d.png' % (opt.outf, epoch),
                                normalize=False)

    # logging
    log_value('loss_d', loss_d.data[0], epoch)
    log_value('loss_g1', loss_g1.data[0], epoch)
    log_value('loss_g2', loss_g2.data[0], epoch)
    log_value('loss_g3', loss_g3.data[0], epoch)
    log_value('loss_e', loss_e.data[0], epoch)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netE.state_dict(), '%s/netE_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
