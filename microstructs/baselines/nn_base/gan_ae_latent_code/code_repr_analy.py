#!/usr/bin/env python
# encoding: utf-8

''' Visualize and analyze latent codes encoded from trained adversarial autoencoders.'''

from __future__ import print_function
import os
import sys
import pdb

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.autograd import Variable
from colorama import Fore

from advae_essentials.args import parser
from advae_essentials.dataloader import multithread_loader, dataloader_retriever
from advae_essentials.model import _senetE

args = parser.parse_args()

# Get latent code-label pairs
# 1. Load trained model
if args.resume: # Resume from saved checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)

        print("=> creating model")
        netE = _senetE()
        if args.cuda:
            netE = netE.cuda()

        netE.load_state_dict(checkpoint['netE_state_dict'])
        print("=> loaded model with checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print( "=> no checkpoint found at '{}'"
              .format( Fore.RED + args.resume + Fore.RESET), file=sys.stderr)
        sys.exit(0)

# 2. Get data-label pairs
sen_ctl = 'all_sentence_constq.ctl'
featpath = '/mnt/data/wenboz/ForenFile_data/sentence_constq_feats'

print('=> loading data ...')
train_loader = multithread_loader(sen_ctl, featpath, num_to_load=args.numLoads, batch_size=args.batchSize,
                                  use_multithreading=False, num_workers=0)

# 3. Encode z
cod_lbl_par = [] # collection of latent code-label pairs
for x, y in train_loader:
    x = Variable( x.cuda(), volatile=True )
    z = netE(x).view(-1)
    print(z.data, y)
    cod_lbl_par.append( (z.data.cpu().numpy(), y.numpy()) )

# Visualize
fig = plt.figure()
ax = fig.add_subplot(111)

px = [ p[0] for p, _ in cod_lbl_par]
py = [ p[1] for p, _ in cod_lbl_par]
ax.scatter(px, py)

fig.savefig('2dim_codes_fm.png')
print('Done.')
