#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import numpy as np
import pdb

def dataloader(ctlpath, phn_list, featpath, target='speaker_id',
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

    for phn in phn_list:
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
                    idx += 1
                    continue

                if speaker in speaker_dict:
                    spk_id = speaker_dict[speaker]
                else:
                    idx += 1
                    continue

                # load data
                spec = np.loadtxt(os.path.join(featpath, p, phn, str(idx)))

                # put data
                spk_id = np.array(spk_id, dtype=int)
                dia_id = np.array(dia_id, dtype=int)
                spec = np.array(spec, dtype=float).reshape((1, 1, 257, -1))
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
