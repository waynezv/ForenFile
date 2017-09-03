#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import numpy as np
import torch
import torch.utils.data as data_utils
from threading import Thread, RLock
from tqdm import tqdm
import time
import pdb

glob_X = []
glob_Y = []
spk_dict = dict()


def sen_dataloader(ctl, featpath):
    '''
    Load features for sentence segments and labels.
    '''
    X = []
    Y = []
    for s in tqdm(ctl, desc='loading', leave=True):
        # TODO: sync dict writing
        spk_name = s.split('/')[2]
        if spk_name in spk_dict:
            spk_id = spk_dict[spk_name]
        else:
            spk_dict[spk_name] = len(spk_dict)
            spk_id = spk_dict[spk_name]

        fn = os.path.join(featpath, s)
        feat = np.loadtxt(fn, delimiter=',')
        X.append( feat )
        Y.append( spk_id )
    return X, Y


class thread_loader(Thread):
    '''
    Data loading operation for each thread.
    '''
    def __init__(self, threadID, lock, loader_func, args):
        super(thread_loader, self).__init__()
        self.threadID = threadID
        self.lock = lock
        self.loader_func = loader_func
        self.args = args

    def run(self):
        print('Start thread: ', self.threadID)
        sub_x, sub_y = self.loader_func(*self.args)
        # sync putting data
        self.lock.acquire()
        glob_X.append(sub_x)
        glob_Y.append(sub_y)
        self.lock.release()
        print('Done thread: ', self.threadID)


def multithread_loader(ctl, featpath, batch_size=64, num_thread=1):
    '''
    Multithread data loading.
    '''
    flist = [ l.rstrip('\n') for l in open(ctl) ]
    flist = flist[:2000]
    num_f = len(flist)
    sub_num_f = num_f // num_thread

    threads_pool = []
    tlock = RLock()
    # create threads
    for i in range(num_thread):
        sub_list = flist[i*sub_num_f: (i+1)*sub_num_f]
        args = (sub_list, featpath)
        threads_pool.append(thread_loader(i, tlock, sen_dataloader, args))

    end = time.time()
    # start threads
    for t in threads_pool:
        t.start()

    # join threads
    for t in threads_pool:
        t.join()

    print('Used {:.3f} s'.format(time.time()-end))

    # batch data
    X = np.asarray(glob_X[0], dtype=float).reshape((-1, 1, 414, 450))
    X = torch.from_numpy(X).float()
    Y = np.asarray(glob_Y[0], dtype=int).reshape((-1))
    Y = torch.from_numpy(Y).int()

    X = (X - X.mean()) / X.std()
    train = data_utils.TensorDataset(X, Y)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=32)

    return train_loader


def phn_dataloader(ctlpath, phn_list, featpath, target='speaker_id',
               min_nframe=5, max_nframe=25, batch=True, batch_size=64, test_batch_size=128,
               shuffle=True):
    # data = dict()
    # data['train'] = []
    # data['test'] = []
    data = []

    phase = ['train', 'test']

    # init dicts
    dialect_dict = {'dr6': 0}
    speaker_dict = {'mcmj0': 0}
    # speaker_dict = {'mjdh0': 0}

    for phn in tqdm(phn_list, desc='loading', leave=True):
        for p in phase:
            # read ctl
            ctlname = os.path.join(ctlpath, p, phn+'.ctl')
            ctls = [ x.split()[0] for x in open(ctlname) ]

            idx = 1
            # load each class label and data in ctl
            for l in ctls:
                x = l.split('/')
                dialect = x[1]
                speaker = x[2]

                if dialect in dialect_dict:
                    dia_id = dialect_dict[dialect]
                else: # only take same speaker
                    dia_id = 1

                if speaker in speaker_dict:
                    spk_id = speaker_dict[speaker]
                else:
                    spk_id = 1

                # load data
                spec = np.loadtxt(os.path.join(featpath, p, phn, str(idx)+'.constq'),
                                  delimiter=',')
                # put data
                spk_id = np.array(spk_id, dtype=int)
                dia_id = np.array(dia_id, dtype=int)
                spec = np.array(spec, dtype=float).reshape((1, 1, 414, -1))
                data.append({'x':spec,
                             'spk_id':spk_id,
                             'dia_id':dia_id})

                idx += 1

    if shuffle:
        data = np.random.permutation(data)

    # batch process
    if batch:
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
                                ).reshape((bs, 1, num_freq, -1))
                mini_y = np.array([ raw[j][target_name]
                                    for j in range(i*bs, (i+1)*bs) ])

                batch_data[p].append( (mini_x, mini_y) )

            x_left = np.array([ raw[j]['x']
                            for j in range(num_batch*bs, len(raw)) ]
                            ).reshape((len(raw) - num_batch*bs, 1, num_freq, -1))
            y_left = np.array([ raw[j][target_name]
                                    for j in range(num_batch*bs, len(raw)) ])

            batch_data[p].append( (x_left, y_left))

    return data
