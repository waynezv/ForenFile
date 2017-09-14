#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import sys
import time
import pdb
import random
import numpy as np
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
from colorama import Fore

from args import parser
from dataloader import multithread_loader, dataloader_retriever
from model import _senetE, _senetG, _senetD, weights_init
from utils import save_checkpoint, ScoreMeter

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

cudnn.benchmark = True

EPS = 1e-12

# load data
ctlpath = './phctl_out'
sen_ctl = 'all_sentence_constq.ctl'
featpath = '/mnt/data/wenboz/ForenFile_data/sentence_constq_feats'

print('=> loading data ...')
train_loader = multithread_loader(sen_ctl, featpath, num_to_load=160, batch_size=args.batchSize)
SPK_ID_INTEREST = 5

# Init model
if args.resume: # Resume from saved checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        old_args = checkpoint['args']
        print('Old args:')
        print(old_args)

        print("=> creating model")
        netD = _senetD()
        netG = _senetG()
        netE = _senetE()
        if args.cuda:
            netD = nn.DataParallel(netD, device_ids=[0]).cuda()
            netG = nn.DataParallel(netG, device_ids=[0]).cuda()
            netE = nn.DataParallel(netE, device_ids=[0]).cuda()

        netD.load_state_dict(checkpoint['netD_state_dict'])
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netE.load_state_dict(checkpoint['netE_state_dict'])
        print("=> loaded model with checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print( "=> no checkpoint found at '{}'"
              .format( Fore.RED + args.resume + Fore.RESET), file=sys.stderr)
        sys.exit(0)

else: # Create model from scratch
    print("=> creating model")
    netD = _senetD()
    netG = _senetG()
    netE = _senetE()
    netD.apply(weights_init)
    netG.apply(weights_init)
    netE.apply(weights_init)
    if args.cuda:
        netD = nn.DataParallel(netD, device_ids=[0]).cuda()
        netG = nn.DataParallel(netG, device_ids=[0]).cuda()
        netE = nn.DataParallel(netE, device_ids=[0]).cuda()

print(netG)
print(netE)
print(netD)

fixed_z = torch.FloatTensor(args.batchSize, 1024, 1, 1).normal_(0, 1)
# label = torch.FloatTensor(1)
# real_label = 1
# fake_label = 0

if args.cuda:
    fixed_z = fixed_z.cuda()

fixed_z = Variable(fixed_z)

if args.eval:
    print("=> evaluating model")
    real_data_interest = dataloader_retriever(train_loader, SPK_ID_INTEREST)
    # vutils.save_image(real_data_interest,
                      # '{}/real_data_interest_{:d}.png'.format(args.outf, SPK_ID_INTEREST),
                      # normalize=False)

    # z1 = Variable(torch.randn(16, 1024, 1, 1).type(torch.cuda.FloatTensor))
    # x1 = netG(z1)

    # z2 = Variable(torch.randn(16, 1024, 1, 1).type(torch.cuda.FloatTensor))
    # x2 = netG(z2)

    # z3= Variable(torch.zeros(16, 1024, 1, 1).type(torch.cuda.FloatTensor))
    # x3 = netG(z3)

    # z4= Variable(torch.ones(16, 1024, 1, 1).type(torch.cuda.FloatTensor))
    # x4 = netG(z4)

    # z5= Variable(torch.rand(16, 1024, 1, 1).type(torch.cuda.FloatTensor))
    # x5 = netG(z5)

    # mu = np.zeros((1024,))
    # var = np.eye(1024)
    # z = np.sort(np.random.multivariate_normal(mu, var, 16), None)
    # z = Variable(torch.from_numpy(z).type(torch.cuda.FloatTensor)).view(16, 1024, 1, 1)
    zs = []
    for i in np.arange(-10, 10.5, 0.3):
        zs.append( i * torch.ones(1024, 1, 1) + 0.2*torch.rand(1024, 1, 1) + 0.2*(torch.rand(1024, 1, 1) - 1) )
    z = torch.stack(zs, 0)
    z = Variable(z.type(torch.cuda.FloatTensor)).view(-1, 1024, 1, 1)
    x = netG(z)
    vutils.save_image(x.data,
                      '{}/fake_gen_normal_tiles_noise2.png'.format(args.outf),
                      normalize=False)

    # vutils.save_image(x1.data,
                      # '{}/fake_gen_normal1.png'.format(args.outf),
                      # normalize=False)
    # vutils.save_image(x2.data,
                      # '{}/fake_gen_normal2.png'.format(args.outf),
                      # normalize=False)
    # vutils.save_image(x3.data,
                      # '{}/fake_gen_zeros.png'.format(args.outf),
                      # normalize=False)
    # vutils.save_image(x4.data,
                      # '{}/fake_gen_ones.png'.format(args.outf),
                      # normalize=False)
    # vutils.save_image(x5.data,
                      # '{}/fake_gen_uniform.png'.format(args.outf),
                      # normalize=False)
    sys.exit(0)

# setup optimizer
optimizer_D = optim.Adam(netD.parameters(), lr=0.1, betas=(args.beta1, 0.999))
optimizer_G = optim.Adam(netG.parameters(), lr=0.1, betas=(args.beta1, 0.999))
optimizer_E = optim.Adam(netE.parameters(), lr=0.1, betas=(args.beta1, 0.999))

optimizer_DG = optim.Adam(itertools.chain(netD.parameters(), netG.parameters()), lr=0.01, betas=(args.beta1, 0.999))
optimizer_GE = optim.Adam(itertools.chain(netG.parameters(), netE.parameters()), lr=0.01, betas=(args.beta1, 0.999))

configure(args.outf)
updt_cnt = 0
loss_d_meter = ScoreMeter()
loss_e_meter = ScoreMeter()
print("=> traning")
for epoch in range(args.niter):
    for itr, (x, y) in enumerate(train_loader):
        netD.zero_grad()
        netG.zero_grad()
        netE.zero_grad()

        real_cpu = x
        x_real = x
        if args.cuda:
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
        loss_g = loss_g3

        # loss for discriminator
        # loss_d = -0.5*( output_real*interest_mask + EPS ).log().mean() - 0.5*( 1 - output_fake + EPS ).log().mean()
        # loss_d = -0.5*( output_real + EPS ).log().mean() - 0.5*( 1 - output_fake + EPS ).log().mean()
        loss_d = -0.25*( output_real + EPS ).log().mean() - 0.25*( 1 - output_fake + EPS ).log().mean() - \
            0.25*( output_fake + EPS ).log().mean() - 0.25*( 1 - output_real + EPS ).log().mean()

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
            epoch, args.niter, itr, len(train_loader)*args.batchSize,
            loss_d.data[0], loss_g.data[0], loss_e.data[0]))

        # Save images
        if (itr % 100) == 0:
            vutils.save_image(real_cpu,
                              '%s/real_samples.png' % args.outf,
                              normalize=False)
            fake = netG(fixed_z)
            vutils.save_image(fake.data,
                              '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                              normalize=False)
            x_reconstruct = netG(netE(x_real))
            vutils.save_image(x_reconstruct.data,
                              '%s/real_reconstruct_epoch_%03d.png' % (args.outf, epoch),
                              normalize=False)

            loss_d_meter.update(loss_d.data[0])
            loss_e_meter.update(loss_e.data[0])

    # logging
    log_value('loss_d', loss_d.data[0], epoch)
    log_value('loss_g1', loss_g1.data[0], epoch)
    log_value('loss_g2', loss_g2.data[0], epoch)
    log_value('loss_g3', loss_g3.data[0], epoch)
    log_value('loss_e', loss_e.data[0], epoch)

    # do checkpointing
    # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
    # torch.save(netE.state_dict(), '%s/netE_epoch_%d.pth' % (args.outf, epoch))
    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
    save_checkpoint({
        'args': args,
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netE_state_dict': netE.state_dict(),
        'netD_state_dict': netD.state_dict()
    }, args.outf, 'checkpoint_epoch_{:d}.pth.tar'.format(epoch))

loss_d_meter.save('loss_d', os.path.join(args.outf, 'loss_d.tsv'))
loss_e_meter.save('loss_e', os.path.join(args.outf, 'loss_e.tsv'))
