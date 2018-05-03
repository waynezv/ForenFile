#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import errno
import sys
import random
import itertools
import time
from tqdm import tqdm
import pdb

import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt

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
    _zpreConditioner, make_onehot, weights_init, calc_gradient_penalty
from advae_essentials.utils import save_checkpoint, ScoreMeter, slerp

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
rng = np.random.RandomState(seed=args.manualSeed)

if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# Check CUDA
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# CUDNN
# cudnn.benchmark = True
cudnn.fastest = True

# Init model
if args.resume:  # Resume from saved checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        old_args = checkpoint['args']
        print('Old args:')
        print(old_args)

        print("=> creating model")
        netE = _senetE().cuda()
        netG = _senetG().cuda()
        netD = _codeNetD().cuda()
        netP = _zpreConditioner(args.topk_class).cuda()

        netE.load_state_dict(checkpoint['netE_state_dict'])
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        netP.load_state_dict(checkpoint['netP_state_dict'])
        print("=> loaded model with checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'"
              .format(Fore.RED + args.resume + Fore.RESET), file=sys.stderr)
        sys.exit(0)

else:  # Create model from scratch
    print("=> creating model")
    netE = _senetE().cuda()
    netG = _senetG().cuda()
    netD = _codeNetD().cuda()
    netP = _zpreConditioner(args.topk_class).cuda()
    # netE = torch.nn.DataParallel(netE, device_ids=[0, 1, 2, 3]).cuda()
    # netG = torch.nn.DataParallel(netG, device_ids=[0, 1, 2, 3]).cuda()
    # netD = torch.nn.DataParallel(netD, device_ids=[0, 1, 2, 3]).cuda()
    # netP = torch.nn.DataParallel(netP, device_ids=[0, 1, 2, 3]).cuda()

    netE.apply(weights_init)
    netG.apply(weights_init)
    netD.apply(weights_init)
    netP.apply(weights_init)


print(netE)
print(netG)
print(netD)
print(netP)

# Load data
sen_ctl = 'all_sentence_constq.ctl'
# featpath = '/mnt/data/wenboz/ForenFile_data/sentence_constq_feats'
featpath = '/home/wenboz/ProJEX/data_root/sentence_constq_feats'

print('=> loading data for task ' + Fore.GREEN + '{}'.format(args.label) + Fore.RESET)
train_loader, test_loader = multithread_loader(sen_ctl, featpath, label=args.label, topk_class=args.topk_class, num_to_load=args.numLoads, batch_size=args.batchSize,
                                               use_multithreading=False, num_workers=0)

# Init variables
y_orac = torch.arange(0, args.topk_class).type(torch.LongTensor)  # oracle labels [0, topk_class)
zpre_orac = make_onehot(y_orac, args.topk_class)  # onehot vectors
one = torch.FloatTensor([1]).cuda()  # gradient for Wasserstein loss
mone = torch.FloatTensor([-1]).cuda()
EPS = 1e-12

# Evaluate model
if args.eval:
    print("=> evaluating model")
    if not os.path.exists(os.path.join(args.outf, 'eval')):
        os.makedirs(os.path.join(args.outf, 'eval'))

    # PCA z
    if 0:
        diter = iter(test_loader)
        x, y = diter.next()
        z = netE(Variable(x.cuda(), volatile=True))

        # PCA z
        z = z.data.view(-1, 200).cpu().numpy()
        z = z - z.mean(axis=0)  # standarize z
        # Covariance matrix
        z_cov = z.T.dot(z) / float(z.shape[0] - 1)  # equiv to using np.cov(zs.T)
        # Eigen decomposition
        evals, evecs = np.linalg.eig(z_cov)  # evecs[:,i] == np.linalg.svd(zs) and take v[i,:]
        eig_pairs = [(np.abs(evals[i]), evecs[:, i]) for i in range(len(evals))]
        eig_pairs.sort()  # sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.reverse()
        # Projection
        P = [ep[1] for ep in eig_pairs]
        P = np.array(P).reshape((-1, 200))  # each row is eigen vector

        # Reconstruct X with z components
        Xs = torch.zeros(5, 414, 450)
        for i in range(5):
            P_ = np.zeros_like(P)
            P_[i, :] = P[i, :]
            z_ = z.dot(P_.T)
            Xs[i, :, :] = netG(Variable(torch.from_numpy(z_).view(-1, 200, 1, 1).float().cuda(), volatile=True)).data[0, :, :, :].view(-1, 414, 450)

        X_topk = torch.zeros(5, 414, 450)
        for k in range(5):
            P_topk = np.zeros_like(P)
            P_topk[0:k, :] = P[0:k, :]
            z_topk = z.dot(P_topk.T)[0]
            X_topk[i, :, :] = netG(Variable(torch.from_numpy(z_topk).view(-1, 200, 1, 1).float().cuda(), volatile=True)).data.view(-1, 414, 450)

        Xs = Xs.cpu().numpy()
        X_topk = X_topk.cpu().numpy()
        savemat(os.path.join(args.outf, 'x_pca_single'), {'x': Xs})
        savemat(os.path.join(args.outf, 'x_pca_compo'), {'x': X_topk})

    # Reconstruct
    if 0:
        diter = iter(test_loader)
        x, y = diter.next()
        pdb.set_trace()
        x_np = x[:4, :, :, :].view(-1, 414, 450).numpy()
        xr = netG(netE(Variable(x[:4, :, :, :].cuda())))
        xr_np = xr.data.view(-1, 414, 450).cpu().numpy()
        savemat(os.path.join(args.outf, 'x_ori_speaker_latest_noS'), {'x': x_np})
        savemat(os.path.join(args.outf, 'x_recon_speaker_latest_noS'), {'x': xr_np})

    # Sampling
    if 0:
        z, m, v = netP(Variable(zpre_orac.cuda(), volatile=True))
        m0 = m[0]
        m1 = m[1]
        m2 = m[2]
        m3 = m[3]

        v0 = v[0]
        v1 = v[1]
        v2 = v[2]
        v3 = v[3]

        z0s = torch.zeros(5, 200)
        z0s[0, :] = z.data[0, :, :, :].view(200)

        z1s = torch.zeros(5, 200)
        z1s[0, :] = z.data[1, :, :, :].view(200)

        z2s = torch.zeros(5, 200)
        z2s[0, :] = z.data[2, :, :, :].view(200)

        z3s = torch.zeros(5, 200)
        z3s[0, :] = z.data[3, :, :, :].view(200)

        for i in range(1, 5):
            z0s[i, :] = torch.normal(m0, v0).data
            z1s[i, :] = torch.normal(m1, v1).data
            z2s[i, :] = torch.normal(m2, v2).data
            z3s[i, :] = torch.normal(m3, v3).data

        x_gen_0 = netG(Variable(z0s.view(-1, 200, 1, 1).cuda()))
        x_gen_1 = netG(Variable(z1s.view(-1, 200, 1, 1).cuda()))
        x_gen_2 = netG(Variable(z2s.view(-1, 200, 1, 1).cuda()))
        x_gen_3 = netG(Variable(z3s.view(-1, 200, 1, 1).cuda()))

        x_gen_0_np = x_gen_0.data.view(-1, 414, 450).cpu().numpy()
        x_gen_1_np = x_gen_1.data.view(-1, 414, 450).cpu().numpy()
        x_gen_2_np = x_gen_2.data.view(-1, 414, 450).cpu().numpy()
        x_gen_3_np = x_gen_3.data.view(-1, 414, 450).cpu().numpy()

        savemat(os.path.join(args.outf, 'x_gen_spk_0'), {'x': x_gen_0_np})
        savemat(os.path.join(args.outf, 'x_gen_spk_1'), {'x': x_gen_1_np})
        savemat(os.path.join(args.outf, 'x_gen_spk_2'), {'x': x_gen_2_np})
        savemat(os.path.join(args.outf, 'x_gen_spk_3'), {'x': x_gen_3_np})

    # Interpolation
    if 1:
        diter = iter(train_loader)
        x, y = diter.next()
        xA = x[0]
        xB = x[1]
        zA = netE(Variable(xA.view(-1, 1, 414, 450).cuda()))
        zB = netE(Variable(xB.view(-1, 1, 414, 450).cuda()))

        # zA, _, _ = netP(Variable(zpre_orac[0].view(1, -1).cuda(), volatile=True))
        # zB, _, _ = netP(Variable(zpre_orac[1].view(1, -1).cuda(), volatile=True))

        zA, zB = zA.data.view(200).cpu().numpy(), zB.data.view(200).cpu().numpy()
        vals = np.linspace(0, 1, )
        zABs = np.zeros((5, 200))
        for i in range(5):
            zABs[i, :] = slerp(vals[i], zA, zB)

        pdb.set_trace()
        xABs = netG(Variable(torch.from_numpy(zABs).view(-1, 200, 1, 1).float().cuda()))

        xABs_np = xABs.data.view(-1, 414, 450).cpu().numpy()
        savemat(os.path.join(args.outf, 'x_AB'), {'x': xABs_np})

    print('Done.')
    sys.exit(0)

# Record settings
if not os.path.exists(os.path.join(args.outf, 'images')):
    os.makedirs(os.path.join(args.outf, 'images'))
if not os.path.exists(os.path.join(args.outf, 'records')):
    os.makedirs(os.path.join(args.outf, 'records'))
configure(os.path.join(args.outf, 'records'))

lossD_meter = ScoreMeter()
errD_real_meter = ScoreMeter()
errD_fake_meter = ScoreMeter()
errD_grad_meter = ScoreMeter()
lossE_meter = ScoreMeter()
lossP_meter = ScoreMeter()
lossG_meter = ScoreMeter()
lossScatter_meter = ScoreMeter()
test_err_meter = ScoreMeter()

# Setup optimizer
# 1e-4, 0.9, 0.999
optimizer_D = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer_E = optim.Adam(netE.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer_P = optim.Adam(netP.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer_G = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.9, 0.999))

# Other training settings
gen_iterations = 0
Diters = 5
errD_grad_penalty = 10
# scatter_y = torch.FloatTensor([-1])
scatter_margin = torch.FloatTensor([args.topk_class]).sqrt_()
num_gens_per_epoch = len(train_loader) // 8
save_freqency = 1 * num_gens_per_epoch  # save every # epoch
if save_freqency < 1:
    save_freqency = 1
print('Save frequency: ', save_freqency)
old_record_fn = 'youll_never_find_me'
best_test_error = 1e19
best_epoch = 0
best_generation = 0

# Train model
print("=> traning")
for epoch in range(args.niter):
    data_iter = iter(train_loader)
    i = 0  # data counter
    while i < len(train_loader):
        ############################
        # Update netG netE
        ############################
        for p in netE.parameters():
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = True
        for p in netD.parameters():
            p.requires_grad = False
        for p in netP.parameters():
            p.requires_grad = False

        x, y = data_iter.next()
        i += 1
        netE.zero_grad()
        netG.zero_grad()

        x_ori = Variable(x.cuda())
        x_reconstruct = netG(netE(x_ori))
        lossG = torch.nn.functional.mse_loss(x_reconstruct, x_ori)
        lossG.backward()

        optimizer_G.step()
        optimizer_E.step()
        gen_iterations += 1

        ############################
        # Update netP
        ############################
        for p in netP.parameters():
            p.requires_grad = True
        for p in netE.parameters():
            p.requires_grad = False
        for p in netD.parameters():
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = False

        if i < len(train_loader):
            x, y = data_iter.next()
            i += 1
        else:  # when data runs out, get new iter
            new_data_iter = iter(train_loader)
            x, y = new_data_iter.next()

        netP.zero_grad()

        zpre = make_onehot(y, args.topk_class)
        _, zm, zvar = netP(Variable(zpre.cuda()))
        # Scatter loss: constrain distance of Gaussians wrt their variances
        # lossScatter = scatter_margin * Sw / Sb
        # Sw = zvar.norm(p=2, dim=1).pow(2).sum()  # within-class scatter
        # Sbm = (zm - zm.mean(dim=0)).norm(p=2, dim=1).pow(2).sum()  # between-class scatter BUG: always zero
        # Sb = zm.norm(p=2, dim=1).pow(2).sum()  # between-class scatter
        # lossScatter = (Sw - 1).norm() + (Sb - 1).norm() + \
            # torch.nn.functional.margin_ranking_loss(Sw, Sb, Variable(scatter_y.cuda()), margin=scatter_margin)
        # Sb = (Variable(scatter_margin.cuda()) - (zm - zm.mean(dim=0)).norm(p=2, dim=1).mean()).norm(p=1)
        Sm = (zm.norm(p=2, dim=1).mean() - Variable(scatter_margin.cuda())).norm(p=1)  # norm to sphere sqrt(C)
        Sv = (zvar.norm(p=2, dim=1) - zm.norm(p=2, dim=1) + Variable(scatter_margin.div(2).cuda())).norm(p=1).sum()  # scatter var relative to mean
        zmn = zm.div(zm.norm(p=2, dim=1).view(-1, 1).repeat(1, 200))  # normalize
        inner_prod = Variable(torch.zeros(zmn.size(0), zmn.size(0)).cuda())
        for zi in range(zmn.size(0) - 1):
            for zj in range(zi + 1, zmn.size(0)):
                inner_prod[zi, zj] = zmn[zi].dot(zmn[zj])
        Sa = inner_prod.norm(p=1).sum()  # angular scatter
        print('Sm {:.4f}  Sv {:.4f}  Sa {:.4f}'.format(Sm.data[0], Sv.data[0], Sa.data[0]))
        lossScatter = Sv
        lossScatter.backward()

        optimizer_P.step()

        ############################
        # Update netD
        ############################
        for p in netD.parameters():
            p.requires_grad = True
        for p in netE.parameters():
            p.requires_grad = False
        for p in netP.parameters():
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = False

        dj = 0  # D update counter
        while (dj < Diters):
            dj += 1
            if i < len(train_loader):
                x, y = data_iter.next()
                i += 1
            else:  # when data runs out, get new iter
                new_data_iter = iter(train_loader)
                x, y = new_data_iter.next()

            netD.zero_grad()

            # Train with real
            x_real = Variable(x.cuda(), volatile=True)  # volatile for inference only
            zr = netE(x_real)
            zrv = Variable(zr.data)
            output_real = netD(zrv)
            errD_real = output_real.mean()
            errD_real.backward(mone)  # gradient for cost ||zf - zr||

            # Train with fake
            zpre = make_onehot(y, args.topk_class)
            zf, _, _ = netP(Variable(zpre.cuda(), volatile=True))
            zfv = Variable(zf.data)
            output_fake = netD(zfv)
            errD_fake = output_fake.mean()
            errD_fake.backward(one)

            # Lipschitz constraint on D
            errD_grad = calc_gradient_penalty(netD, zrv.data, zfv.data, penalty=errD_grad_penalty)  # gradient penalty
            errD_grad.backward()
            lossD = errD_fake - errD_real + errD_grad

            optimizer_D.step()

        ############################
        # Update netE netP
        ############################
        for p in netE.parameters():
            p.requires_grad = True
        for p in netP.parameters():
            p.requires_grad = True
        for p in netD.parameters():
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = False

        if i < len(train_loader):
            x, y = data_iter.next()
            i += 1
        else:  # when data runs out, get new iter
            new_data_iter = iter(train_loader)
            x, y = new_data_iter.next()

        netE.zero_grad()
        netP.zero_grad()

        x_real = Variable(x.cuda())
        output_real = netD(netE(x_real))
        lossE = output_real.mean()
        lossE.backward(one)

        zpre = make_onehot(y, args.topk_class)
        zf, _, _ = netP(Variable(zpre.cuda()))
        output_fake = netD(zf)
        lossP = output_fake.mean()
        lossP.backward(mone)

        optimizer_E.step()
        optimizer_P.step()

        print('[{:d}/{:d}][{:d}/{:d}][{:d}] '.format(epoch, args.niter, i, len(train_loader), gen_iterations) +
              Fore.RED + 'LossD: {:.4f} '.format(lossD.data[0]) + Fore.RESET +
              'ErrD_real: {:.4f} ErrD_fake: {:.4f} ErrD_grad: {:.4f} '.format(errD_real.data[0], errD_fake.data[0], errD_grad.data[0]) +
              Fore.BLUE + 'LossE: {:.4f} LossP: {:.4f} LossScatter: {:.4f} '.format(lossE.data[0], lossP.data[0], lossScatter.data[0]) + Fore.RESET +
              Fore.GREEN + 'LossG: {:.4f}'.format(lossG.data[0]) + Fore.RESET)

        # Save images & test
        if (gen_iterations % save_freqency) == 0:  # every 5 epochs
            vutils.save_image(x_real.data.cpu(),
                              '{}/real_samples_gen_{:03d}_epoch_{:d}.png'.format(os.path.join(args.outf, 'images'), gen_iterations, epoch),
                              normalize=False)

            x_reconstruct = netG(netE(x_real))
            vutils.save_image(x_reconstruct.data,
                              '{}/real_reconstruct_gen_{:03d}_epoch_{:d}.png'.format(os.path.join(args.outf, 'images'), gen_iterations, epoch),
                              normalize=False)

            fake = netG(zf)
            vutils.save_image(fake.data,
                              '{}/fake_samples_gen_{:03d}_epoch_{:d}.png'.format(os.path.join(args.outf, 'images'), gen_iterations, epoch),
                              normalize=False)

            # Test
            # end_timer = time.time()
            # trues = []  # true labels
            # preds = []  # predicted labels
            # _, zo_mean, zo_var = netP(Variable(zpre_orac.cuda(), volatile=True))
            # zo_mean = zo_mean.view(-1, 200)
            # zo_var = zo_var.view(-1, 200)
            # for x, y in tqdm(test_loader, desc='testing', leave=True):  # iterate over batches
                # trues.append(y.numpy())
                # z_test = netE(Variable(x.cuda(), volatile=True)).view(-1, 200)
                # dists = Variable(torch.zeros(z_test.size(0), zo_mean.size(0)).cuda())
                # for ti, zi in enumerate(z_test):
                    # for tj, (zomj, zovarj) in enumerate(zip(zo_mean, zo_var)):
                        # dists[ti, tj] = ((zi - zomj).pow(2).div(zovarj.pow(2))).sum()
                # _, mini = dists.topk(1, dim=1, largest=False, sorted=True)  # find min index
                # # # Compute prob
                # # pred_j = []  # preds for zs (in a batch)
                # # for zi in z:
                    # # prob_i = []  # probs for zi
                    # # for m in means:  # iterate over mixtures
                        # # p = stats.multivariate_normal.pdf(zi.view(-1).data.cpu().numpy(), m, variance)
                        # # prob_i.append(p)
                    # # pred_j.append(np.argmax(prob_i))  # NOTE: max prob!
                # preds.append(mini.data.cpu().numpy())
            # # Compute accuracy
            # trues = np.array(list(itertools.chain.from_iterable(trues))).reshape((-1,))
            # preds = np.array(list(itertools.chain.from_iterable(preds))).reshape((-1,))
            # pred_error = 1. - np.count_nonzero(trues == preds) / float(len(trues))
            # print('Trues: ', trues)
            # print('Preds: ', preds)
            # print('Test error: {:.4f}'.format(pred_error))
            # print('Test time: {:.4f}s'.format(time.time() - end_timer))

            # # Save best
            # is_best = pred_error < best_test_error
            # if is_best:
                # best_test_error = pred_error
                # best_epoch = epoch
                # best_generation = gen_iterations
                # save_checkpoint({
                    # 'args': args,
                    # 'epoch': epoch,
                    # 'best_epoch': best_epoch,
                    # 'gen_iterations': gen_iterations,
                    # 'best_generation': best_generation,
                    # 'best_test_error': best_test_error,
                    # 'netE_state_dict': netE.state_dict(),
                    # 'netG_state_dict': netG.state_dict(),
                    # 'netD_state_dict': netD.state_dict(),
                    # 'netP_state_dict': netP.state_dict()
                # }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_BEST.pth.tar')
                # print(Fore.GREEN + 'Saved checkpoint for best test error {:.4f} at epoch {:d}'.format(best_test_error, best_epoch) + Fore.RESET)

            # Logging
            # log_value('test_err', pred_error, gen_iterations)
            log_value('lossD', lossD.data[0], gen_iterations)
            log_value('errD_real', errD_real.data[0], gen_iterations)
            log_value('errD_fake', errD_fake.data[0], gen_iterations)
            log_value('errD_grad', errD_grad.data[0], gen_iterations)
            log_value('lossE', lossE.data[0], gen_iterations)
            log_value('lossP', lossP.data[0], gen_iterations)
            log_value('lossG', lossG.data[0], gen_iterations)
            log_value('lossScatter', lossScatter.data[0], gen_iterations)
            # test_err_meter.update(pred_error)
            lossD_meter.update(lossD.data[0])
            errD_real_meter.update(errD_real.data[0])
            errD_fake_meter.update(errD_fake.data[0])
            errD_grad_meter.update(errD_grad.data[0])
            lossE_meter.update(lossE.data[0])
            lossP_meter.update(lossP.data[0])
            lossG_meter.update(lossG.data[0])
            lossScatter_meter.update(lossScatter.data[0])

            # Checkpointing
            save_checkpoint({
                'args': args,
                'epoch': epoch,
                'gen_iterations': gen_iterations,
                'netE_state_dict': netE.state_dict(),
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'netP_state_dict': netP.state_dict()
            }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_gen_{:d}_epoch_{:d}.pth.tar'.format(gen_iterations, epoch))

            # Delete old checkpoint to save space
            new_record_fn = os.path.join(args.outf, 'checkpoints', 'checkpoint_gen_{:d}_epoch_{:d}.pth.tar'.format(gen_iterations, epoch))
            if os.path.exists(old_record_fn) and os.path.exists(new_record_fn):
                os.remove(old_record_fn)
            old_record_fn = new_record_fn

test_err_meter.save('test_err', os.path.join(args.outf, 'records'), 'test_err.tsv')
lossD_meter.save('lossD', os.path.join(args.outf, 'records'), 'lossD.tsv')
errD_real_meter.save('errD_real', os.path.join(args.outf, 'records'), 'errD_real.tsv')
errD_fake_meter.save('errD_fake', os.path.join(args.outf, 'records'), 'errD_fake.tsv')
errD_grad_meter.save('errD_grad', os.path.join(args.outf, 'records'), 'errD_grad.tsv')
lossE_meter.save('lossE', os.path.join(args.outf, 'records'), 'lossE.tsv')
lossP_meter.save('lossP', os.path.join(args.outf, 'records'), 'lossP.tsv')
lossG_meter.save('lossG', os.path.join(args.outf, 'records'), 'lossG.tsv')
lossScatter_meter.save('lossScatter', os.path.join(args.outf, 'records'), 'lossScatter.tsv')
