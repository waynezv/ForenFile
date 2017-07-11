#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import sys
import os
import re
import argparse

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

def dataloader(ctlpath, phn, featpath):
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
            if (spec.ndim == 1) or (spec.shape[1] <= 5): # few frames
                idx = idx + 1
                continue

            # put data
            # spk_id_vec = np.zeros((630,), dtype=int)
            # spk_id_vec[spk_id] = 1
            # dia_id_vec = np.zeros((8,), dtype=int)
            # dia_id_vec[dia_id] = 1
            spk_id = np.array(spk_id, dtype=int)
            dia_id = np.array(dia_id, dtype=int)
            data[p].append({'x':spec,
                            'spk_id':spk_id,
                            'dia_id':dia_id})

    return data['train'], data['test']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.conv2 = nn.Conv2d(3, 5, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        # self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(1024, 630)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2_drop(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        l = x.size(1)
        x = F.relu(nn.Linear(l, 1024)(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def train(dtrain, model, optimizer, epoch, args):
    # Train mode
    model.train()
    batch_idx = 0
    end = time.time()
    for idx, record in enumerate(dtrain):
        # Load data
        data = record['x']
        target = record['spk_id']
        h, w = data.shape
        data = data.reshape((1,1,h,w))
        target = target.reshape((-1,))
        data = torch.from_numpy(data).float()
        # target = torch.from_numpy(target).float()
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
                  .format( epoch, batch_idx * len(data), len(dtrain),
                          100. * batch_idx / len(dtrain), loss.data[0],
                          batch_duration ))

        # Log
        log_value('train loss', loss, idx)

        batch_idx += 1

def test(dtest, model, optimizer, epoch, args):
    model.eval()
    test_loss = 0
    correct = 0
    end = time.time()
    for idx, record in enumerate(dtest):
        # Load data
        data = record['x']
        target = record['spk_id']
        h, w = data.shape
        data = data.reshape((1,1,h,w))
        target = target.reshape((-1,))
        data = torch.from_numpy(data).float()
        # target = torch.from_numpy(target).float()
        target = torch.LongTensor(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # Forward
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader) # loss function already averages over batch size
    batch_duration = time.time() - end
    end = time.time()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time: {:.3f}\n'
          .format( test_loss, correct, len(dtest), 100. * correct / len(dtest),
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
    phn = 'AA'
    featpath = './phsegwav_feat'

    dtrain, dtest = dataloader(ctlpath, phn, featpath)

    # Define model
    model = Net()
    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=[]).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    cudnn.benchmark = True # for acceleration, but may bring overhead
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
        train(dtrain, model, optimizer, epoch)
        test_loss = test(dtest, model, optimizer, epoch)

        # Save best
        # TODO
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


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
    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.cuda = True

    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # train_loader = torch.utils.data.DataLoader(
        # datasets.MNIST('../data', train=True, download=True,
                    # transform=transforms.Compose([
                        # transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,)) ])
                    # ),
        # batch_size=args.batch_size, shuffle=True, **kwargs)

    main(args)
