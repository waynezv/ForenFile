#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os, errno
import sys
import pdb
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from tensorboard_logger import configure, log_value
from colorama import Fore

from args import parser
from dataloader import multithread_loader, dataloader_retriever
from model import _codeNetE, _codeNetG, _codeNetD, _senetE, _senetG, \
    weights_init, calc_gradient_penalty
from utils import save_checkpoint, ScoreMeter, sample_GMM, onehot_categorical

# Parse args
args = parser.parse_args()
print(args)

# Make dirs
try:
    os.makedirs(args.outf)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Fix seed for randomization
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
rng = np.random.RandomState(seed = args.manualSeed)

if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# Check CUDA
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

cudnn.benchmark = True

# Init model
if args.resume: # Resume from saved checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        old_args = checkpoint['args']
        print('Old args:')
        print(old_args)

        print("=> creating model")
        netE = _senetE()
        netG = _senetG()
        netD = _codeNetD()
        if args.cuda:
            netE = netE.cuda()
            netG = netG.cuda()
            netD = netD.cuda()

        netE.load_state_dict(checkpoint['netE_state_dict'])
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        print("=> loaded model with checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print( "=> no checkpoint found at '{}'"
              .format( Fore.RED + args.resume + Fore.RESET), file=sys.stderr)
        sys.exit(0)

else: # Create model from scratch
    print("=> creating model")
    netE = _senetE()
    netG = _senetG()
    netD = _codeNetD()

    netE.apply(weights_init)
    netG.apply(weights_init)
    netD.apply(weights_init)

    if args.cuda:
        netE = netE.cuda()
        netG = netG.cuda()
        netD = netD.cuda()

print(netE)
print(netG)
print(netD)

# Load data
ctlpath = './phctl_out'
sen_ctl = 'all_sentence_constq.ctl'
featpath = '/mnt/data/wenboz/ForenFile_data/sentence_constq_feats'

print('=> loading data ...')
train_loader = multithread_loader(sen_ctl, featpath, num_to_load=args.numLoads, batch_size=args.batchSize)

# Init variables
fixed_z = torch.FloatTensor(args.batchSize, 3, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1
EPS = 1e-12

if args.cuda:
    fixed_z = fixed_z.cuda()
    one, mone = one.cuda(), mone.cuda()

fixed_z = Variable(fixed_z)

# Evaluate model
if args.eval:
    print("=> evaluating model")
    if not os.path.exists(os.path.join(args.outf, 'eval')):
        os.makedirs(os.path.join(args.outf, 'eval'))

    print('Done.')
    sys.exit(0)

# Setup optimizer
optimizer_E = optim.Adam(netE.parameters(), lr=0.0001, betas=(args.beta1, 0.999))
optimizer_G = optim.Adam(netG.parameters(), lr=0.0001, betas=(args.beta1, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=0.0001, betas=(args.beta1, 0.999))
# optimizer_D = optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)
# Other training settings
if not os.path.exists(os.path.join(args.outf, 'images')):
    os.makedirs(os.path.join(args.outf, 'images'))
if not os.path.exists(os.path.join(args.outf, 'records')):
    os.makedirs(os.path.join(args.outf, 'records'))
configure(os.path.join(args.outf, 'records'))

loss_d_meter = ScoreMeter()
loss_dr_meter = ScoreMeter()
loss_df_meter = ScoreMeter()
loss_e_meter = ScoreMeter()
loss_g_meter = ScoreMeter()

# Train model
print("=> traning")
gen_iterations = 0
Diters = 5
for epoch in range(args.niter):
    data_iter = iter(train_loader)
    i = 0 # data counter
    while i < len(train_loader):
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netE update
        for p in netE.parameters():
            p.requires_grad = False

        # Train the discriminator Diters times
        j = 0 # D update counter
        while (j < Diters) and (i < len(train_loader)) :
            j += 1
            ############################
            # (1) Update netD
            ############################
            x, _ = data_iter.next()
            i += 1

            # Train with real
            x_real_cpu = x # torch tensor on cpu
            x_real = x

            if args.cuda:
                x_real = x_real.cuda()

            x_real = Variable(x_real, volatile=True) # volatile for inference only

            netD.zero_grad()
            zr = netE(x_real)
            zrv = Variable(zr.data)
            output_real = netD(zrv)

            errD_real = output_real.mean()
            errD_real.backward(mone)

            # Train with fake
            zf = torch.FloatTensor(x_real.size(0), 3, 1, 1).normal_(0, 1)
            if args.cuda:
                zf = zf.cuda()
            zf = Variable(zf)
            output_fake = netD(zf)
            errD_fake = output_fake.mean()
            errD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, zr.data, zf.data)
            gradient_penalty.backward()
            errD = errD_real - errD_fake + gradient_penalty

            # errD = errD_real - errD_fake
            optimizer_D.step()

        ############################
        # (2) Update netE
        ############################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        for p in netE.parameters():
            p.requires_grad = True

        if i < len(train_loader):
            x, _ = data_iter.next()
            i += 1
        else:
            new_data_iter = iter(train_loader)
            x, _ = new_data_iter.next()

        x_real_cpu = x # torch tensor on cpu
        x_real = x
        if args.cuda:
            x_real = x_real.cuda()
        x_real = Variable(x_real)

        netE.zero_grad()
        zr = netE(x_real)
        output_real = netD(zr)
        errE = output_real.mean()
        errE.backward(one)
        optimizer_E.step()

        ############################
        # (3) Update netG
        ############################
        for p in netE.parameters():
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = True

        if i < len(train_loader):
            x, _ = data_iter.next()
            i += 1
        else:
            new_data_iter = iter(train_loader)
            x, _ = new_data_iter.next()

        x_real_cpu = x # torch tensor on cpu
        x_real = x
        if args.cuda:
            x_real = x_real.cuda()
        x_real = Variable(x_real)

        netG.zero_grad()
        zr = netE(x_real)
        zrv = Variable(zr.data)
        x_reconstruct = netG(zrv)
        errG = (x_reconstruct - x_real).pow(2).mean()

        errG.backward()
        optimizer_G.step()

        gen_iterations += 1

        print('[{:d}/{:d}][{:d}/{:d}][{:d}] Loss_D: {:.4f} Loss_D_real: {:.4f} Loss_D_fake: {:.4f} Loss_E: {:.4f} Loss_G: {:.4f}'.format(
            epoch, args.niter, i, len(train_loader), gen_iterations,
            errD.data[0], errD_real.data[0], errD_fake.data[0], errE.data[0], errG.data[0]))

        # Save images
        if (gen_iterations % 100) == 0:
            vutils.save_image(x_real_cpu,
                              '{}/real_samples_gen_{:03d}.png'.format(os.path.join(args.outf, 'images'), gen_iterations),
                              normalize=False)

            vutils.save_image(x_reconstruct.data,
                              '{}/real_reconstruct_gen_{:03d}.png'.format(os.path.join(args.outf, 'images'), gen_iterations),
                              normalize=False)

            zf = torch.FloatTensor(x_real.size(0), 3, 1, 1).normal_(0, 1)
            if args.cuda:
                zf = zf.cuda()
            zf = Variable(zf, volatile=True)
            fake = netG(zf)
            vutils.save_image(fake.data,
                              '{}/fake_samples_gen_{:03d}.png'.format(os.path.join(args.outf, 'images'), gen_iterations),
                              normalize=False)


            # Logging
            log_value('loss_d', errD.data[0], gen_iterations)
            log_value('loss_dr', errD_real.data[0], gen_iterations)
            log_value('loss_df', errD_fake.data[0], gen_iterations)
            log_value('loss_grad_d', gradient_penalty.data[0], gen_iterations)
            log_value('loss_e', errE.data[0], gen_iterations)
            log_value('loss_g', errG.data[0], gen_iterations)
            loss_d_meter.update(errD.data[0])
            loss_dr_meter.update(errD_real.data[0])
            loss_df_meter.update(errD_fake.data[0])
            loss_e_meter.update(errE.data[0])
            loss_g_meter.update(errG.data[0])

            # Checkpointing
            save_checkpoint({
                'args': args,
                'epoch': epoch,
                'gen_iterations': gen_iterations,
                'netE_state_dict': netE.state_dict(),
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict()
            }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_gen_{:d}_epoch_{:d}.pth.tar'.format(gen_iterations, epoch))

loss_d_meter.save('loss_d', os.path.join(args.outf, 'records'), 'loss_d.tsv')
loss_dr_meter.save('loss_dr', os.path.join(args.outf, 'records'), 'loss_dr.tsv')
loss_df_meter.save('loss_df', os.path.join(args.outf, 'records'), 'loss_df.tsv')
loss_e_meter.save('loss_e', os.path.join(args.outf, 'records'), 'loss_e.tsv')
loss_g_meter.save('loss_g', os.path.join(args.outf, 'records'), 'loss_g.tsv')
