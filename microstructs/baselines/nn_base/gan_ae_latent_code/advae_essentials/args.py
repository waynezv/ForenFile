#!/usr/bin/env python
# encoding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='timit | tidigits | to be added')
parser.add_argument('--dataroot', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)

parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

parser.add_argument('--resume', default='', help='path containing saved models to resume from')
parser.add_argument('--eval', action='store_true', help='evaluate trained model')

parser.add_argument('--label', default='speaker', help='target label to train and evaluate')
parser.add_argument('--topk_class', type=int, default=100, help='top k classes to load')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--numLoads', type=int, help='number of training samples to load')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
