#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import numpy as np

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
