#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import numpy as np
import torch
import torch.utils.data as data_utils
from threading import Thread, RLock
import itertools
from tqdm import tqdm
import time
import pdb

glob_X = []  # global container for all feature Xs
glob_Y = []  # global container for all label Ys
spk_dict = dict()  # dictionary {speaker name: speaker id}
dia_dict = dict()  # dictionary {speaker name: dialect id}
gender_dict = {'f': 0, 'm': 1}  # dictionary for male or female
label_count = dict()  # {label_id: count}


def save_speaker_dict(fn):
    '''
    Save speaker dictionary.
    '''
    out = "speaker\tid\n"
    for k in spk_dict:
        out += "{}\t{}\n".format(k, spk_dict[k])

    with open(fn, 'w') as f:
        f.write(out)


def dataloader_retriever(dataloader, spk_id):
    '''
    Retrieve data for specific spk id.
    :dataloader (DataLoader)
    :spk_id (int)
    <- retrieved data (Torch Tensor)
    '''
    x_id = []  # store feature (x)s for speaker (id)
    for x, y in iter(dataloader):
        id_idx = y.eq(spk_id).nonzero().numpy()
        if id_idx.size != 0:
            x = x[id_idx, :, :, :].view(-1, 414, 450)
            x_id.append([x[i, :, :].numpy() for i in range(x.size(0))])

    x_id = list(itertools.chain.from_iterable(x_id))
    x_id = torch.from_numpy(np.array(x_id)).view(-1, 1, 414, 450)

    y_id = torch.LongTensor(x_id.size(0)).fill_(spk_id)
    return x_id, y_id


def sen_dataloader(ctl, featpath, label='speaker'):
    '''
    Load features for sentence segments and labels.
    '''
    X = []  # store feature Xs
    Y = []  # store label Ys
    for s in tqdm(ctl, desc='loading', leave=True):
        spk_name = s.split('/')[2]
        dia = s.split('/')[1]

        if label == 'speaker':
            if spk_name in spk_dict:
                spk_id = spk_dict[spk_name]
                label_count[spk_id] += 1
            else:
                spk_dict[spk_name] = len(spk_dict)
                spk_id = spk_dict[spk_name]
                label_count[spk_id] = 1
            ly = spk_id

        elif label == 'gender':
            gender = spk_name[0]  # get female or male
            gender_id = gender_dict[gender]  # only two classes so use predefined dict
            if gender_id in label_count:
                label_count[gender_id] += 1
            else:
                label_count[gender_id] = 1
            ly = gender_id

        elif label == 'dialect':
            if dia in dia_dict:
                dia_id = dia_dict[dia]
                label_count[dia_id] += 1
            else:
                dia_dict[dia] = len(dia_dict)
                dia_id = dia_dict[dia]
                label_count[dia_id] = 1
            ly = dia_id

        fn = os.path.join(featpath, s)
        feat = np.loadtxt(fn, delimiter=',')
        X.append(feat)
        Y.append(ly)
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
        # Get subset of Xs and Ys in each thread
        sub_x, sub_y = self.loader_func(*self.args)
        # Sync putting data
        self.lock.acquire()
        glob_X.append(sub_x)
        glob_Y.append(sub_y)
        self.lock.release()
        print('Done thread: ', self.threadID)


def multithread_loader(ctl, featpath, label='speaker', topk_class=100, num_to_load=None, batch_size=64, use_multithreading=False, num_thread=1, num_workers=1, shuffle=True):
    '''
    Multithread data loading.
    '''
    # load data
    flist = [l.rstrip('\n') for l in open(ctl)]
    if num_to_load:
        flist = flist[:num_to_load]

    if use_multithreading == False:  # if not using multithread
        sX, sY = sen_dataloader(flist, featpath, label)
        X = np.asarray(sX, dtype=float).reshape((-1, 1, 414, 450))
        Y = np.asarray(sY, dtype=int).reshape((-1))

    elif use_multithreading == True:  # if use multithread
        num_f = len(flist)
        sub_num_f = num_f // num_thread

        threads_pool = []
        tlock = RLock()
        # create threads
        for i in range(num_thread):
            sub_list = flist[i * sub_num_f: (i + 1) * sub_num_f]
            args = (sub_list, featpath, label)
            threads_pool.append(thread_loader(i, tlock, sen_dataloader, args))

        end = time.time()
        # start threads
        for t in threads_pool:
            t.start()

        # join threads
        for t in threads_pool:
            t.join()

        print('Used {:.3f} s'.format(time.time() - end))

        X = np.asarray(glob_X[0], dtype=float).reshape((-1, 1, 414, 450))
        Y = np.asarray(glob_Y[0], dtype=int).reshape((-1))

    # normalize data
    X = (X - X.mean()) / X.std()  # center

    # split data
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    split_ratio = 0.8
    gcount = 0  # count total samples
    topk_count = 0  # count topk classes
    for s in label_count:  # get sample count for each label
        if (gcount >= Y.shape[0]) or (topk_count >= topk_class):
            break
        count = label_count[s]
        train_count = int(np.floor(count * split_ratio))
        sub_cnt = 0
        while sub_cnt < train_count:
            Xtrain.append(X[gcount])
            Ytrain.append(Y[gcount])
            sub_cnt += 1
            gcount += 1
        while sub_cnt < count:
            Xtest.append(X[gcount])
            Ytest.append(Y[gcount])
            sub_cnt += 1
            gcount += 1
        topk_count += 1

    print('Loaded total {:d} classes with {:d} samples; train {:d} test {:d}'.format(topk_count, gcount, len(Ytrain), len(Ytest)))

    # save speaker dict
    # save_speaker_dict('speaker_dict.ctl')

    # batch data
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    Xtrain = torch.from_numpy(Xtrain).float()
    Ytrain = torch.from_numpy(Ytrain).int()
    Xtest = torch.from_numpy(Xtest).float()
    Ytest = torch.from_numpy(Ytest).int()

    train = data_utils.TensorDataset(Xtrain, Ytrain)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test = data_utils.TensorDataset(Xtest, Ytest)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, test_loader


def phn_dataloader(ctlpath, phn_list, featpath, target='speaker_id',

                   min_nframe=5, max_nframe=25, batch=True, batch_size=64, test_batch_size=128, shuffle=True):
    data = []
    phase = ['train', 'test']

    # init dicts
    dialect_dict = {'dr6': 0}
    speaker_dict = {'mcmj0': 0}
    # speaker_dict = {'mjdh0': 0}

    for phn in tqdm(phn_list, desc='loading', leave=True):
        for p in phase:
            # read ctl
            ctlname = os.path.join(ctlpath, p, phn + '.ctl')
            ctls = [x.split()[0] for x in open(ctlname)]

            idx = 1
            # load each class label and data in ctl
            for l in ctls:
                x = l.split('/')
                dialect = x[1]
                speaker = x[2]

                if dialect in dialect_dict:
                    dia_id = dialect_dict[dialect]
                else:  # only take same speaker
                    dia_id = 1

                if speaker in speaker_dict:
                    spk_id = speaker_dict[speaker]
                else:
                    spk_id = 1

                # load data
                spec = np.loadtxt(os.path.join(featpath, p, phn, str(idx) + '.constq'),
                                  delimiter=',')
                # put data
                spk_id = np.array(spk_id, dtype=int)
                dia_id = np.array(dia_id, dtype=int)
                spec = np.array(spec, dtype=float).reshape((1, 1, 414, -1))
                data.append({'x': spec,
                             'spk_id': spk_id,
                             'dia_id': dia_id})

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
                mini_x = np.array([raw[j]['x']
                                   for j in range(i * bs, (i + 1) * bs)]
                                  ).reshape((bs, 1, num_freq, -1))
                mini_y = np.array([raw[j][target_name]
                                   for j in range(i * bs, (i + 1) * bs)])

                batch_data[p].append((mini_x, mini_y))

            x_left = np.array([raw[j]['x']
                               for j in range(num_batch * bs, len(raw))]
                            ).reshape((len(raw) - num_batch * bs, 1, num_freq, -1))
            y_left = np.array([raw[j][target_name]
                               for j in range(num_batch * bs, len(raw))])

            batch_data[p].append((x_left, y_left))

    return data
