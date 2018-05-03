from __future__ import print_function
import sys
import time
import os
import shutil
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

from colorama import Fore


def slerp(val, low, high):
    '''
    Spherical linear interpolation. val has a range of 0 to 1.
    '''
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low

    omega = np.arccos(np.dot(low, high) / (np.linalg.norm(high) * np.linalg.norm(high)))
    so = np.sin(omega)
    # return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high
    return (1.0 - val) * low + val * high


def gen_means_slerp(mean_l, mean_h, C):
    '''
    Return C means between [mean_l, mean_h] via slerp.
    '''
    vals = np.linspace(0, 1, C)
    means = [slerp(v, mean_l, mean_h) for v in vals]
    return means


class sample_GMM():
    '''
    GMM sampler.
    '''
    def __init__(self, num_dim, num_mixtures, rng):
        self.num_dim = num_dim
        self.num_mixtures = num_mixtures
        self.rng = rng

    def _gen_mean_variance(self):
        self.means = [i * np.ones((self.num_dim, ))
                      for i in range(-(self.num_mixtures // 2), (self.num_mixtures // 2))]
        self.variances = [np.eye(self.num_dim) for _ in range(self.num_mixtures)]

    def _sample_prior(self, num_samples):
        return self.rng.choice(a=self.num_mixtures,
                               size=(num_samples,),
                               replace=True)

    def _sample_gaussian(self, mean, variance):
        epsilons = self.rng.normal(size=(self.num_dim, ))
        return mean + np.linalg.cholesky(variance).dot(epsilons)

    def sample(self, num_samples, labels=None):
        samples = []

        if labels is None:
            priors = self._sample_prior(num_samples).tolist()
        else:
            priors = labels.tolist()

        self._gen_mean_variance()

        for p in priors:
            samples.append(self._sample_gaussian(self.means[p],
                                                 self.variances[p]))
        return np.array(samples), np.array(priors)


def onehot_categorical(num_samples, num_labels, rng):
    '''
    Generate onehot categorical samples.
    '''
    y = np.zeros((num_samples, num_labels), dtype=np.float32)
    indices = rng.randint(0, num_labels, num_samples)

    for i in xrange(num_samples):
                y[i, indices[i]] = 1
    return y


def create_save_folder(save_path, force=False, ignore_patterns=[]):
    '''
    Create new folder and backup old folder.
    '''
    if os.path.exists(save_path):
        print(Fore.RED + save_path + Fore.RESET +
              ' already exists!', file=sys.stderr)
        if not force:
            ans = input('Do you want to overwrite it? [y/N]:')
            if ans not in ('y', 'Y', 'yes', 'Yes'):
                os.exit(1)
        from getpass import getuser
        tmp_path = '/tmp/{}-experiments/{}_{}'.format(getuser(),
                                                      os.path.basename(save_path),
                                                      time.time())
        print('move existing {} to {}'.format(save_path, Fore.RED +
                                              tmp_path + Fore.RESET))
        shutil.copytree(save_path, tmp_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print('create folder: ' + Fore.GREEN + save_path + Fore.RESET)

    # copy code to save folder
    if save_path.find('debug') < 0:
        shutil.copytree('.', os.path.join(save_path, 'src'), symlinks=True,
                        ignore=shutil.ignore_patterns('*.pyc', '__pycache__',
                                                      '*.path.tar', '*.pth',
                                                      '*.ipynb', '.*', 'data',
                                                      'save', 'save_backup',
                                                      save_path,
                                                      *ignore_patterns))


def save_checkpoint(state, save_dir, filename, is_best=False):
    '''
    Save training checkpoints.
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


class ScoreMeter():
    '''
    Record, measure, save scores.
    '''
    def __init__(self):
        self.score = []

    def update(self, val):
        self.score.append(val)

    def save(self, score_name, save_dir, fn):
        scores = "idx\t{}".format(score_name)
        for i, s in enumerate(self.score):
            scores += "\n"
            scores += "{:d}\t{:.4f}".format(i, s)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fn = os.path.join(save_dir, fn)
        with open(fn, 'w') as f:
            f.write(scores)


class AverageMeter(object):
    '''
    Computes and stores the average and current value.
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(model, args):
    '''
    Return optimizer.
    '''
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), args.lr,
                                beta=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
    '''
    Decay Learning rate at 1/2 and 3/4 of the num_epochs.
    '''
    lr = lr_init
    if epoch >= num_epochs * 0.75:
        lr *= decay_rate**2
    elif epoch >= num_epochs * 0.5:
        lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def error_topk(output, target, topk=(1,)):
    '''
    Computes the error@k for the specified values of k.
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res


def error(output, target, topk=(1,)):
    '''
    Compute prediction error.
    '''
    _, pred = output.topk(1)
    correct = pred.eq(target.long())
    res = 100.0 - correct.float().sum() * (100.0 / target.size(0))

    return res


def accuracy(output, target, topk=(1,)):
    '''
    Compute prediction accuracy.
    '''
    _, pred = output.topk(1)
    correct = pred.eq(target.long())

    return correct.float().sum() * (100.0 / target.size(0))


def show_images(imgs):
    '''
    Show grid of images.
    '''
    img_grid = make_grid(imgs, normalize=True, padding=100)
    plt.imshow(np.transpose(img_grid, (1, 2, 0)), interpolation='nearest')
