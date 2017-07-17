#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import sys
import os
import re
import argparse
import time
import shutil

import numpy as np
import scipy.signal as signal

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

# BUG: cudnn/cublas runtime error.
# Fixed by setup cuda and cudnn, and then install Anaconda
# and dependencies, and build pytorch from source.
# cudnn.enabled=False

def dataloader(ctlpath, phn, featpath, target='speaker_id',
               min_nframe=5, max_nframe=10, batch_size=64, test_batch_size=128,
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
        for l in ctls:
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
            x = np.zeros((spec.shape[0], max_nframe), dtype='float')
            x[:, :spec.shape[1]] = spec

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
        for i in range(num_batch):
            mini_x = np.array([ raw[j]['x']
                               for j in range(i*bs, (i+1)*bs) ]
                              ).reshape((bs, 1, -1, max_nframe))
            mini_y = np.array([ raw[j][target_name]
                                for j in range(i*bs, (i+1)*bs) ])

            batch_data[p].append( (mini_x, mini_y) )

        x_left = np.array([ raw[j]['x']
                        for j in range(num_batch*bs, len(raw)) ]
                        ).reshape((-1, 1, num_freq, max_nframe))
        y_left = np.array([ raw[j][target_name]
                                for j in range(num_batch*bs, len(raw)) ])

        batch_data[p].append( (x_left, y_left))

    return batch_data['train'], batch_data['test']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3)
        # self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 630)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        flat_dim = self._flat_dim(x)
        x = x.view(-1, flat_dim)

        m = nn.Linear(flat_dim, 2048).cuda()
        x = F.relu(m(x))

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc2(x)
        x = self.fc3(x)

        return F.log_softmax(x)

    def _flat_dim(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        return np.prod(size)

def train(dtrain, model, optimizer, epoch, args):
    # Train mode
    model.train() # train mode for dropout, BN, etc
    end = time.time()
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

        # Log
        log_value('train loss', loss.data[0], batch_idx)

        batch_idx += 1

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

    # Log

    return test_loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main(args):
    # Load data
    ctlpath = '../../feat_falign_extract/phctl_out'
    phn = 'S'
    featpath = './phsegwav_feat'


    kwargs = {'target': 'speaker_id',
              'min_nframe': 5, 'max_nframe': 10,
              'batch_size': args.batch_size, 'test_batch_size': args.test_batch_size,
              'shuffle': True}

    print('Loading data ...')
    end = time.time()
    dtrain, dtest = dataloader(ctlpath, phn, featpath, **kwargs)
    print('Done. Time: {:.3f}'.format(time.time()-end))

    # Define model
    model = Net()
    model
    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        # model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    cudnn.benchmark = False # for acceleration, but may bring overhead
    cudnn.fastest = False

    # Optionally reload model or resume from previous training
    if args.reload is True:
        pass

    # Optionally evaluate
    if args.evaluate is True:
        pass

    # Train, test and log
    configure("runs/aa_spk_0710")
    best_loss = 1000000
    for epoch in range(1, args.epochs + 1):
        train(dtrain, model, optimizer, epoch, args)
        test_loss = test(dtest, model, optimizer, epoch, args)

        # Save best
        # TODO
        # is_best = test_loss < best_loss
        # best_loss = min(test_loss, best_loss)
        # save_checkpoint({
            # 'epoch': epoch + 1,
            # 'arch': args.arch,
            # 'state_dict': model.state_dict(),
            # 'best_prec1': best_prec1,
            # 'optimizer' : optimizer.state_dict(),
        # }, is_best)


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
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
