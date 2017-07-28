#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pdb
import os
import sys
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
from more_itertools import unique_everseen
import numpy as np
import skimage
from skimage import io

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from colorama import Fore
from importlib import import_module

import voc_utils as vutil
import spatial_mlp as smlp
from spatial_mlp_trainer import Trainer
import config
from utils import save_checkpoint, get_optimizer, create_save_folder
from args import arg_parser, arch_resume_names
# TODO
# import make_graph as mk_grf

try:
    from tensorboard_logger import configure, log_value
except BaseException:
    configure = None

# Dataset
root_dir = '/home/wenboz/ProJEX/data_root/VOCdevkit/VOC2012'
img_dir = os.path.join(root_dir, 'JPEGImages')
ann_dir = os.path.join(root_dir, 'Annotations')
set_dir = os.path.join(root_dir, 'ImageSets', 'Main')

img_set_cat = vutil.list_image_sets()
num_cat = len(img_set_cat)

CLASS = img_set_cat[13]
print('Object to detect: ', CLASS)

# Load data
# TODO: load less val
# TODO: multithreading
# TODO: use DataLoader Class
def dataloader(batch_size):
# data list
    trn_img_fn = [ vutil.imgs_from_category_as_list(c, 'train')
                for c in img_set_cat ]

    val_img_fn = [ vutil.imgs_from_category_as_list(c, 'val')
                for c in img_set_cat ]

    trn_ls = [ trn_img_fn[i][j] for i in range(num_cat)
            for j in range(trn_img_fn[i].size) ]

    val_ls = [ val_img_fn[i][j] for i in range(num_cat)
            for j in range(val_img_fn[i].size) ]

# shuffle
    trn_ls = np.random.permutation(trn_ls)
    val_ls = np.random.permutation(val_ls)

# labels
    trn_lbl = []
    for img in tqdm(trn_ls, desc='processing train', leave=True):
        anno = vutil.load_annotation(img)
        names = anno.find_all(text=CLASS)
        class_label = 1 if names else 0
        trn_lbl.append(class_label)

    val_lbl = []
    for img in tqdm(val_ls, desc='processing val', leave=True):
        anno = vutil.load_annotation(img)
        names = anno.find_all(text=CLASS)
        class_label = 1 if names else 0
        val_lbl.append(class_label)

    def get_batch_ls(ls, batch_size):
        bls = []
        num_tot = len(ls)
        num_b = num_tot // batch_size

        for i in tqdm(range(num_b), desc='batching', leave=True):
            bls.append( [ ls[j] for j in range(i*batch_size, (i+1)*batch_size) ] )

        bls.append( ls[num_b*batch_size:] )

        return num_b, bls

# batch
    _, btrn_ls = get_batch_ls(trn_ls, batch_size)
    _, btrn_lbl = get_batch_ls(trn_lbl, batch_size)
    _, bval_ls = get_batch_ls(val_ls, batch_size)
    _, bval_lbl = get_batch_ls(val_lbl, batch_size)

    dtrn = zip( btrn_ls, btrn_lbl )
    dval = zip( bval_ls, bval_lbl )

    return dtrn, dval


def main():
    global args
    best_loss = 1.e12
    best_epoch = 0

    args = arg_parser.parse_args()
    args.config_of_data = config.datasets[args.data]
    args.num_classes = config.datasets[args.data]['num_classes']

    if configure is None:
        args.tensorboard = False
        print(Fore.RED +
              'WARNING: you don\'t have tesnorboard_logger installed' +
              Fore.RESET)

    # optionally resume from a checkpoint
    if args.resume:
        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            old_args = checkpoint['args']
            print('Old args:')
            print(old_args)
            # set args based on checkpoint
            # TODO: necessary?
            if args.start_epoch <= 0:
                args.start_epoch = checkpoint['epoch'] + 1
            best_epoch = checkpoint['best_epoch']
            best_loss = checkpoint['best_loss']
            for name in arch_resume_names:
                if name in vars(args) and name in vars(old_args):
                    setattr(args, name, getattr(old_args, name))
            # TODO
            # model = getModel(**vars(args))
            print("=> creating model")
            model = smlp.Spatial_MLP()
            model = nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
            print(model)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        else:
            print( "=> no checkpoint found at '{}'"
                  .format( Fore.RED + args.resume + Fore.RESET),
                  file=sys.stderr)
            return

    else:
        # create model
        print("=> creating model")
        model = smlp.Spatial_MLP()
        model = nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
        print(model)

    # define loss function (criterion)
    criterion = nn.BCEWithLogitsLoss().cuda()

    # define optimizer
    optimizer = get_optimizer(model, args)

    # Trainer
    trainer = Trainer(model, criterion, optimizer, args)

    # Load data
    print("=> loading data")
    dtrain, dval = dataloader(args.batch_size)

    # Evaluate
    stt = time.time()
    if args.evaluate == 'train':
        print("=> evaluating model for training set from epoch {}".format(best_epoch))
        train_loss, train_err = trainer.test(dtrain, best_epoch)
        print('Done in {:.3f}s'.format(time.time()-stt))
        return
    elif args.evaluate == 'val':
        print("=> evaluating model for testing set from epoch {}".format(best_epoch))
        val_loss, val_err = trainer.test(dval, best_epoch)
        print('Done in {:.3f}s'.format(time.time()-stt))
        return

    # check if the folder exists
    create_save_folder(args.save, args.force)

    # set up logging
    global log_print, f_log
    f_log = open(os.path.join(args.save, 'log.txt'), 'w')

    def log_print(*args):
        print(*args)
        print(*args, file=f_log)

    log_print('args:')
    log_print(args)
    print('model:', file=f_log)
    print(model, file=f_log)
    log_print('# of params:',
              str(sum([p.numel() for p in model.parameters()])))

    f_log.flush()
    torch.save(args, os.path.join(args.save, 'args.pth'))
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_err1'
              '\tval_err1\ttrain_err5\tval_err']

    if args.tensorboard:
        configure(args.save, flush_secs=5)

    print("=> training")
    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):

        # train for one epoch
        train_loss, lr = trainer.train(
            dtrain, epoch)

        if args.tensorboard:
            log_value('lr', lr, epoch)
            log_value('train_loss', train_loss, epoch)

        # evaluate on validation set
        val_loss, val_err = trainer.test(dval, epoch)

        if args.tensorboard:
            log_value('val_loss', val_loss, epoch)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{}' + '\t{:.4f}' * 2)
                      .format(epoch, lr, train_loss, val_loss))
        with open(os.path.join(args.save, 'scores.tsv'), 'w') as f:
            print('\n'.join(scores), file=f)

        # remember best err@1 and save checkpoint
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_epoch = epoch
            print(Fore.GREEN + 'Best var_err1 {}'.format(best_loss) +
                  Fore.RESET)
            # test_loss, test_err1, test_err1 = validate(
            #     test_loader, model, criterion, epoch, True)
            # save test
        save_checkpoint({
            'args': args,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best, args.save)
        if not is_best and epoch - best_epoch >= args.patience > 0:
            break

    print('Best val_loss: {:.4f} at epoch {}'.format(best_loss, best_epoch))


if __name__ == '__main__':
    main()
