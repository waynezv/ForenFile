#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import argparse
import time
import shutil
from colorama import Fore
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from tensorboard_logger import configure, log_value

from tqdm import tqdm

from spatial_mlp import Spatial_MLP as SMLP
from utils import create_save_folder, save_checkpoint

# BUG: cudnn/cublas runtime error.
# Fixed by setup cuda and cudnn, and then install Anaconda
# and dependencies, and build pytorch from source.
# cudnn.enabled=False

def dataloader(ctlpath, phn, featpath, target='speaker_id',
               min_nframe=5, max_nframe=10, batch_size=64, test_batch_size=128,
               pad=True,
               shuffle=True):
    data = dict()
    data['train'] = []
    data['test'] = []

    phase = ['train', 'test']

    for p in phase:
        # read ctl
        ctlname = os.path.join(ctlpath, p, phn+'.ctl')
        ctls = [ x.split()[0] for x in open(ctlname) ]

        # init dicts
        dialect_dict = dict()
        speaker_dict = dict()

        idx = 1
        # load each class label and data in ctl
        for l in tqdm(ctls, desc='Extracting', leave=True):
            x = l.split('/')
            dialect = x[1]
            speaker = x[2]
            # encode class labels
            if dialect in dialect_dict:
                dia_id = dialect_dict[dialect]
            else:
                dialect_dict[dialect] = len(dialect_dict)
                dia_id = dialect_dict[dialect]

            if speaker in speaker_dict:
                spk_id = speaker_dict[speaker]
            else:
                speaker_dict[speaker] = len(speaker_dict)
                spk_id = speaker_dict[speaker]

            # load data
            spec = np.loadtxt(os.path.join(featpath, p, phn, str(idx)))
            if (spec.ndim == 1) or (spec.shape[1] < min_nframe) or (spec.shape[1] > max_nframe):
                idx = idx + 1
                continue
            # pad
            if pad:
                x = np.zeros((spec.shape[0], max_nframe), dtype='float')
                x[:, :spec.shape[1]] = spec
            else:
                x = np.array(spec, dtype='float')

            # put data
            spk_id = np.array(spk_id, dtype=int)
            dia_id = np.array(dia_id, dtype=int)
            data[p].append({'x':x,
                            'spk_id':spk_id,
                            'dia_id':dia_id})
        if shuffle:
            data[p] = np.random.permutation(data[p])

    # batch process
    target_name = 'spk_id' if target=='speaker_id' else 'dia_id'
    batch_dict = {'train': batch_size, 'test': test_batch_size}
    batch_data = dict()
    for p in phase:
        batch_data[p] = []
        raw = data[p]
        bs = batch_dict[p]
        num_batch = len(raw) // bs
        num_freq = raw[0]['x'].shape[0]
        for i in tqdm(range(num_batch), desc='Batching', leave=True):
            mini_x = np.array([ raw[j]['x']
                               for j in range(i*bs, (i+1)*bs) ]
                              ).reshape((bs, 1, num_freq, -1))
            mini_y = np.array([ raw[j][target_name]
                                for j in range(i*bs, (i+1)*bs) ])

            batch_data[p].append( (mini_x, mini_y) )

        num_left = len(raw) - num_batch*bs
        x_left = np.array([ raw[j]['x']
                        for j in range(num_batch*bs, len(raw)) ]
                        ).reshape((num_left, 1, num_freq, -1))
        y_left = np.array([ raw[j][target_name]
                                for j in range(num_batch*bs, len(raw)) ])

        batch_data[p].append( (x_left, y_left))

    return batch_data['train'], batch_data['test']

def train(dtrain, model, optimizer, epoch, args):
    # Train mode
    model.train() # train mode for dropout, BN, etc
    end = time.time()
    train_loss = 0
    batch_idx = 0
    for data, target in dtrain:
        # Load data
        data = torch.from_numpy(data).float()
        target = torch.LongTensor(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # Forward
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.data[0]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_duration = time.time() - end
        end = time.time()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\tTime: {:.3f}'
                  .format( epoch, batch_idx * len(data), len(dtrain) * len(data),
                          100. * batch_idx / len(dtrain), loss.data[0],
                          batch_duration ))

        batch_idx += 1

    train_loss /= len(dtrain) # loss function already averages over batch size

    return train_loss


def test(dtest, model, optimizer, epoch, args):
    model.eval()
    test_loss = 0
    correct = 0
    end = time.time()
    for data, target in dtest:
        # Load data
        data = torch.from_numpy(data).float()
        target = torch.LongTensor(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # Forward
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    # TODO
    test_loss /= len(dtest) # loss function already averages over batch size
    batch_duration = time.time() - end
    end = time.time()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time: {:.3f}\n'
          .format( test_loss, correct, len(dtest)*len(data),
                  100. * correct / (len(dtest)*len(data)),
                  batch_duration ))

    return test_loss


def main(args):
    # Load data
    ctlpath = '../../feat_falign_extract/phctl_out'
    phn = 'S'
    featpath = './phsegwav_feat'


    kwargs = {'target': 'speaker_id',
              'min_nframe': 5, 'max_nframe': 10,
              'batch_size': args.batch_size, 'test_batch_size': args.test_batch_size,
              'pad': False,
              'shuffle': True}

    print('Loading data ...')
    end = time.time()
    dtrain, dtest = dataloader(ctlpath, phn, featpath, **kwargs)
    print('Done in {:.3f}s'.format(time.time()-end))

    # Define model
    # model = Net()
    model = SMLP()
    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    cudnn.benchmark = True # for acceleration, but may bring overhead
    cudnn.fastest = False

    # Optionally reload model or resume from previous training
    # TODO
    if args.reload is True:
        pass

    # Optionally evaluate
    # TODO
    if args.evaluate is True:
        pass

    create_save_folder(args.save)

    # Train, test and log
    print('=> training')
    configure(args.save)
    best_loss = 1000000
    for epoch in range(1, args.epochs + 1):
        train_loss = train(dtrain, model, optimizer, epoch, args)
        test_loss = test(dtest, model, optimizer, epoch, args)

        # Log
        log_value('train loss', train_loss, epoch)
        log_value('test loss', test_loss, epoch)


        # Save best
        is_best = test_loss < best_loss
        if is_best:
            best_loss = test_loss
            best_epoch = epoch
            print(Fore.GREEN + 'Best test_loss {}'.format(best_loss) +
                  Fore.RESET)

        save_checkpoint({
            'args': args,
            'epoch': epoch + 1,
            'best_epoch': best_epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Classification tasks on TIMIT')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                                            help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                                            help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                            help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                                            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                            help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                                            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                                            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                            help='how many batches to wait before logging training status')
    parser.add_argument('--reload', default=False, metavar='T or F',
                        help='reload or resume from previous model')
    parser.add_argument('--evaluate', default=False, metavar='T or F',
                        help='evaluate trained model')
    parser.add_argument('--save', default='save/default-{}'.format(time.time()),
                        type=str, metavar='SAVE',
                        help='path to the experiment logging directory'
                        '(default: save/debug)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
