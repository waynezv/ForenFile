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

from args import parser
from dataloader import phn_dataloader
from model import _phnetE, _phnetG, _phnetD, weights_init

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
phn_list = ('AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH',
            'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH',
            'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG',
            'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH',
            'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH')
featpath = '../phsegwav_out_constq_feats'

kwargs = {'target': 'speaker_id',
            # 'min_nframe': , 'max_nframe': ,
          'batch': False, 'batch_size': opt.batchSize, 'test_batch_size': opt.batchSize,
            'shuffle': True}

print('Loading data ...')
end = time.time()
dtrain = phn_dataloader(ctlpath, phn_list, featpath, **kwargs)
print('Done. Time: {:.3f}'.format(time.time()-end))

# Init model
netD = _phnetD()
netG = _phnetG()
netE = _phnetE()
netD.apply(weights_init)
netG.apply(weights_init)
netE.apply(weights_init)

criterion = nn.BCELoss()

fixed_z = torch.FloatTensor(1, 64, 16, 4).normal_(0, 1)
label = torch.FloatTensor(1)
real_label = 1
fake_label = 0

if opt.cuda:
    netD = nn.DataParallel(netD, device_ids=[0,1,2]).cuda()
    netG = nn.DataParallel(netG, device_ids=[0,1,2]).cuda()
    netE = nn.DataParallel(netE, device_ids=[0,1,2]).cuda()
    criterion.cuda()
    fixed_z = fixed_z.cuda()

fixed_z = Variable(fixed_z)

# setup optimizer
optimizer_D = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_GE = optim.Adam(itertools.chain(netG.parameters(), netE.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))


for epoch in range(opt.niter):
    for itr, data in enumerate(dtrain):
        x = data['x']
        real_label = data['spk_id']
        netD.zero_grad()
        netG.zero_grad()
        netE.zero_grad()

        z_s = Variable(torch.randn(1, 64, 16, 4).type(torch.cuda.FloatTensor))

        real_cpu = torch.from_numpy(x).float()
        x_real = torch.from_numpy(x).float()
        if opt.cuda:
            x_real = x_real.cuda()
        x_real = Variable(x_real)

        x_fake = netG(z_s, x_real)
        z_r = netE(x_real)
        x_reconstruct = netG(z_r, x_real)

        output_fake = netD(x_fake, z_s)
        output_real = netD(x_real, z_r)

        # loss & back propagation
        if real_label == 0: # target speaker
            loss_d = -torch.mean(torch.log(output_real+EPS) + torch.log(1-output_fake+EPS))
            loss_ge = -torch.mean(torch.log(output_fake+EPS) + torch.log(1-output_real+EPS))
        elif real_label == 1: # other speaker
            loss_d = -torch.mean(torch.log(1-output_real+EPS) + torch.log(1-output_fake+EPS))
            loss_ge = -torch.mean(torch.log(output_fake+EPS) + torch.log(output_real+EPS))
        loss_ae = -torch.mean( torch.norm(x_reconstruct - x_real) + 1.*torch.norm(z_r) )
        loss_2 = loss_ge + loss_ae

        loss_d.backward(retain_graph=True)
        optimizer_D.step()
        loss_2.backward()
        optimizer_GE.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_GE: %.4f loss_AE: %.4f D(G(z), z): %.4f D(x, E(x)): %.4f'
              % (epoch, opt.niter, itr, len(dtrain)*opt.batchSize,
                 loss_d.data[0], loss_ge.data[0], loss_ae.data[0], output_fake.data[0], output_real.data[0]))

        if (itr % 100) == 0:
              vutils.save_image(real_cpu,
                                '%s/real_samples.png' % opt.outf,
                                normalize=True)
              fake = netG(fixed_z, x_real)
              vutils.save_image(fake.data,
                                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                                normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
