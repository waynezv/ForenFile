#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os, errno
import sys
import random
import itertools
import pdb

import numpy as np
from scipy import stats

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable

from tensorboard_logger import configure, log_value
from colorama import Fore

from advae_essentials.args import parser
from advae_essentials.dataloader import multithread_loader
from advae_essentials.model_large_code import _codeNetD, _senetE, _senetG, \
    weights_init, calc_gradient_penalty
from advae_essentials.utils import save_checkpoint, ScoreMeter

# Generate code priors
mean_l = np.linspace(1, 0, 200)
mean_h = np.linspace(0, 1, 200)
variance = np.eye(200)

def slerp(val, low, high):
    '''Spherical interpolation. val has a range of 0 to 1.'''
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low

    omega = np.arccos(np.dot( low/np.linalg.norm(low), high/np.linalg.norm(high) ))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

vals = np.linspace(0, 1, 630)
means = [ slerp(v, mean_l, mean_h) for v in vals ]

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
sen_ctl = 'all_sentence_constq.ctl'
featpath = '/mnt/data/wenboz/ForenFile_data/sentence_constq_feats'

print('=> loading data ...')
train_loader, test_loader = multithread_loader(sen_ctl, featpath, num_to_load=args.numLoads, batch_size=args.batchSize,
                                  use_multithreading=False, num_workers=0)

# Init variables
fixed_z = torch.FloatTensor(args.batchSize, 2, 1, 1).normal_(0, 1)
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

    # PCA z
    if 1:
        zs = [] # collection of zs
        for x,y in train_loader:
            z = netE(Variable(x.cuda(), volatile=True))
            zs.append(z.view(-1, 200).data.cpu().numpy())
        # PCA z
        zs = list(itertools.chain.from_iterable(zs))
        zs = np.array(zs).reshape((-1, 200))
        zs = zs - zs.mean(axis=0) # standarize z
        # Covariance matrix
        z_cov = zs.T.dot(zs) / float(zs.shape[0]-1) # equiv to using np.cov(zs.T)
        # Eigen decomposition
        evals, evecs = np.linalg.eig(z_cov) # evecs[:,i] == np.linalg.svd(zs) and take v[i,:]
        eig_pairs = [( np.abs(evals[i]), evecs[:,i] ) for i in range(len(evals))]
        eig_pairs.sort() # sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.reverse()
        # Projection
        P = [ep[1] for ep in eig_pairs]
        P = np.array(P).reshape((-1, 200)) # each row is eigen vector
        zsp_full = zs.dot(P.T)
        # Reconstruct X with z components
        X_zfull = netG(Variable(torch.from_numpy(zsp_full).view(-1, 200, 1, 1).float().cuda(), volatile=True))
        for k in range(20):
            P_topk = np.zeros_like(P)
            P_topk[0:k, :] = P[0:k, :]
            zsp_topk = zs.dot(P_topk.T)
            X_ztopk = netG(Variable(torch.from_numpy(zsp_topk).view(-1, 200, 1, 1).float().cuda(), volatile=True))
            vutils.save_image(X_ztopk.data,
                              '{}/reconstructed_samples_ztop{:d}.png'.format(os.path.join(args.outf, 'eval'), k+1),
                              normalize=False)
        X_normal = netG(Variable(torch.from_numpy(zs).view(-1, 200, 1, 1).float().cuda(), volatile=True))
        X_original = netG(z)
        vutils.save_image(x,
                          '{}/real_samples.png'.format(os.path.join(args.outf, 'eval')),
                          normalize=False)
        vutils.save_image(X_zfull.data,
                          '{}/reconstructed_samples_zfull.png'.format(os.path.join(args.outf, 'eval')),
                          normalize=False)
        vutils.save_image(X_normal.data,
                          '{}/reconstructed_samples_znormalized.png'.format(os.path.join(args.outf, 'eval')),
                          normalize=False)
        vutils.save_image(X_original.data,
                          '{}/reconstructed_samples_zoriginal.png'.format(os.path.join(args.outf, 'eval')),
                          normalize=False)


    # Classify test
    if 0:
        trues = [] # true labels
        probs = [] # probs
        preds = [] # predicted labels
        for x, y in train_loader: # iterate over batches
            trues.append(y.numpy())
            z = netE(Variable(x.cuda(), volatile=True))
            # Compute prob
            prob_j = [] # probs for zs (in a batch) and all mixtures
            pred_j = [] # preds for zs (in a batch)
            for zi in z:
                zi = zi.view(-1).data.cpu().numpy()
                prob_i = [] # probs for zi
                for m in means: # iterate over mixtures
                    p = stats.multivariate_normal.pdf(zi, m, variance)
                    prob_i.append(p)
                prob_j.append(prob_i)
                pred_j.append(np.argmax(prob_i))
            probs.append(prob_j)
            preds.append(pred_j)

        # Compute accuracy
        probs = list(itertools.chain.from_iterable(probs))
        trues = list(itertools.chain.from_iterable(trues))
        preds = list(itertools.chain.from_iterable(preds))
        pred_error = 1. - np.count_nonzero(trues == preds) / float(len(trues))
        print('Error: {:.4f}'.format(pred_error))

    print('Done.')
    sys.exit(0)

# Setup optimizer
optimizer_E = optim.Adam(netE.parameters(), lr=0.00002, betas=(0.9, 0.999))
optimizer_G = optim.Adam(netG.parameters(), lr=0.00002, betas=(0.9, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.9, 0.999))

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
test_err_meter = ScoreMeter()

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
            x, y = data_iter.next()
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
            # Uncomment for multiclass priors
            zf = torch.zeros(x_real.size(0), 200, 1, 1)
            zfi = 0
            for yi in y:
                mu = means[yi] # pick mean according to label
                zf[zfi, :, :, :] = torch.from_numpy( rng.multivariate_normal(mu, variance) ).float().view(200,1,1)
                zfi += 1

            # Uncomment for gender priors
            # zf = torch.zeros(x_real.size(0), 2, 1, 1)
            # f_ind = y.eq(0).nonzero()
            # for fi in f_ind:
                # zf[fi, :, :, :] = torch.zeros(2,1,1).normal_(-1,1)
            # m_ind = y.eq(1).nonzero()
            # for mi in m_ind:
                # zf[mi, :, :, :] = torch.zeros(2,1,1).normal_(1,1)

            if args.cuda:
                zf = zf.cuda()
            zf = Variable(zf)
            output_fake = netD(zf)
            errD_fake = output_fake.mean()
            errD_fake.backward(one)
            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, zr.data, zf.data, penalty=5)
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
            x, y = data_iter.next()
            i += 1
        else: # when data runs out, get new iter
            new_data_iter = iter(train_loader)
            x, y = new_data_iter.next()

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

        # if i < len(train_loader):
            # x, y = data_iter.next()
            # i += 1
        # else:
            # new_data_iter = iter(train_loader)
            # x, y = new_data_iter.next()

        # x_real_cpu = x # torch tensor on cpu
        # x_real = x
        # if args.cuda:
            # x_real = x_real.cuda()
        # x_real = Variable(x_real)

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

        # Save images & test
        if (gen_iterations % 200) == 0: # ~ 10 epoch
            vutils.save_image(x_real_cpu,
                              '{}/real_samples_gen_{:03d}.png'.format(os.path.join(args.outf, 'images'), gen_iterations),
                              normalize=False)

            vutils.save_image(x_reconstruct.data,
                              '{}/real_reconstruct_gen_{:03d}.png'.format(os.path.join(args.outf, 'images'), gen_iterations),
                              normalize=False)

            zf = torch.zeros(x_real.size(0), 200, 1, 1)
            zfi = 0
            for yi in y:
                mu = means[yi] # pick mean according to label
                zf[zfi, :, :, :] = torch.from_numpy( rng.multivariate_normal(mu, variance) ).float().view(200,1,1)
                zfi += 1

            # zf = torch.zeros(x_real.size(0), 2, 1, 1)
            # f_ind = y.eq(0).nonzero()
            # for fi in f_ind:
                # zf[fi, :, :, :] = torch.zeros(2,1,1).normal_(-1,1)
            # m_ind = y.eq(1).nonzero()
            # for mi in m_ind:
                # zf[mi, :, :, :] = torch.zeros(2,1,1).normal_(1,1)

            if args.cuda:
                zf = zf.cuda()
            zf = Variable(zf, volatile=True)
            fake = netG(zf)
            vutils.save_image(fake.data,
                              '{}/fake_samples_gen_{:03d}.png'.format(os.path.join(args.outf, 'images'), gen_iterations),
                              normalize=False)

            # Test
            trues = [] # true labels
            preds = [] # predicted labels
            for x, y in test_loader: # iterate over batches
                trues.append(y.numpy())
                z = netE(Variable(x.cuda(), volatile=True))
                # Compute prob
                pred_j = [] # preds for zs (in a batch)
                for zi in z:
                    prob_i = [] # probs for zi
                    for m in means: # iterate over mixtures
                        p = stats.multivariate_normal.pdf(zi.view(-1).data.cpu().numpy(), m, variance)
                        prob_i.append(p)
                    pred_j.append(np.argmax(prob_i))
                preds.append(pred_j)
            # Compute accuracy
            trues = list(itertools.chain.from_iterable(trues))
            preds = list(itertools.chain.from_iterable(preds))
            pred_error = 1. - np.count_nonzero(trues == preds) / float(len(trues))
            print('Test error: {:.4f}'.format(pred_error))

            # Logging
            log_value('test_err', pred_error, gen_iterations)
            log_value('loss_d', errD.data[0], gen_iterations)
            log_value('loss_dr', errD_real.data[0], gen_iterations)
            log_value('loss_df', errD_fake.data[0], gen_iterations)
            log_value('loss_grad_d', gradient_penalty.data[0], gen_iterations)
            log_value('loss_e', errE.data[0], gen_iterations)
            log_value('loss_g', errG.data[0], gen_iterations)
            test_err_meter.update(pred_error)
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

test_err_meter.save('test_err', os.path.join(args.outf, 'records'), 'test_err.tsv')
loss_d_meter.save('loss_d', os.path.join(args.outf, 'records'), 'loss_d.tsv')
loss_dr_meter.save('loss_dr', os.path.join(args.outf, 'records'), 'loss_dr.tsv')
loss_df_meter.save('loss_df', os.path.join(args.outf, 'records'), 'loss_df.tsv')
loss_e_meter.save('loss_e', os.path.join(args.outf, 'records'), 'loss_e.tsv')
loss_g_meter.save('loss_g', os.path.join(args.outf, 'records'), 'loss_g.tsv')
