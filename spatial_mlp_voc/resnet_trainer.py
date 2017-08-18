from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from utils import AverageMeter, adjust_learning_rate, error, accuracy
import time
from tqdm import tqdm
import pdb

import voc_utils as vutil


class Trainer(object):
    def __init__(self, model, criterion=None, optimizer=None, args=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args

    def train(self, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        # top1 = AverageMeter()

        # switch to train mode
        self.model.train()

        lr = adjust_learning_rate(self.optimizer, self.args.lr,
                                  self.args.decay_rate, epoch,
                                  self.args.epochs)  # TODO: add custom
        print('Epoch {:3d} lr = {:.6e}'.format(epoch, lr))

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # load images on fly
            # imgs = vutil.load_imgs(input, False)
            imgs = vutil.load_imgs(input, True, (500,500)).reshape(
                (-1, 3, 500, 500))
            input = torch.from_numpy(imgs).float()
            # target = torch.LongTensor(target)
            target = torch.FloatTensor(target).view(-1, 1)

            input = input.cuda(async=True)
            target = target.cuda(async=True)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure error and record loss
            # err1 = error(output.data, target, topk=(1,))
            losses.update(loss.data[0], input.size(0))
            # top1.update(err1, input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.args.print_freq > 0 and \
                    (i + 1) % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      # 'Err@1 {top1.val:.4f}\t'
                      .format(
                          epoch, i + 1, len(train_loader),
                          batch_time=batch_time, data_time=data_time,
                          loss=losses))
                          # top1=top1))

                print('Epoch: {:3d} Train loss {loss.avg:.4f} '
                      # 'Err@1 {top1.avg:.4f}'
                      .format(epoch, loss=losses))
        return losses.avg, lr

    def test(self, val_loader, epoch, silence=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        errors = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        acc_rate = 0.
        for i, (input, target) in tqdm(enumerate(val_loader), desc='evaluating', leave=True):
            # imgs = vutil.load_imgs(input, False)
            imgs = vutil.load_imgs(input, True, (500,500)).reshape(
                (-1, 3, 500, 500))
            input = torch.from_numpy(imgs).float()
            target = torch.FloatTensor(target).view(-1, 1)

            input= input.cuda(async=True)
            target = target.cuda(async=True)

            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)
            err = error(output, target_var)
            acc  = accuracy(output, target_var)

            # measure error and record loss
            errors.update(err.data[0], input.size(0))
            losses.update(loss.data[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('[{}]\tloss {:.4f}\terror {:.4f}\taccuracy {:.4f}'
                  .format(i, loss.data[0], err.data[0], acc.data[0]))

            acc_rate += acc.data[0]

        acc_rate /= len(val_loader)
        if not silence:
            print('accuracy {:.4f}%'.format(acc_rate))
            print('Epoch: {:3d}\tval loss {loss.avg:.4f}\t'
                  'error {err.avg:.4f}%'
                  .format(epoch, loss=losses, err=errors))

        return losses.avg, err
