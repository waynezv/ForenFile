#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import scipy.io.wavfile as wavfile
import numpy as np
import pdb

base_path = '/media/sdb/dataset/timit/timit'
wav_list_path = '../lists/timit_all.ctl'
save_path = 'sentence_segments'

wav_list = [ l.rstrip('\n') for l in open(wav_list_path) ]

for w in wav_list:
    fs, wd = wavfile.read( os.path.join(base_path, w+'.wav') )

    # convert to float
    nb_bits = 16
    max_nb_bits = float(2 ** (nb_bits - 1))
    wd = wd / (max_nb_bits + 1.0)

    if wd.size >= 32000: # duration 2s
        wd = wd[:32000]

        savefn = os.path.join(save_path, w+'_2s.wav')
        h, t = os.path.split(savefn)
        if not os.path.exists(h):
            os.makedirs(h)

        wavfile.write(savefn, fs, wd)
        print('wrote {0:s}'.format(savefn))
