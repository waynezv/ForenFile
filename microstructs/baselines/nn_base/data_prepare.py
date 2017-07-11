#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import sys
import os
import re

import numpy as np
import scipy.io as sio
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def compute_feat(wavctl, wavpath, outpath, feat):
    wavs = [ x.rstrip('\n') for x in open(wavctl) ]
    for w in wavs:
        fs, wd = wavfile.read(os.path.join(wavpath, w))

        # scale to -1.0 -- 1.0
        nb_bits = 16 # 16-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        wd = wd / (max_nb_bit + 1.0)

        # compute feat, currently spectrogram only
        f, t, spec = signal.spectrogram(wd, fs=fs,
                                    window='hamming',
                                    nperseg=256,
                                    noverlap=256//8,
                                    nfft=512,
                                    scaling='density')

# sio.savemat('test_spec.mat', {'f':f, 't':t, 'spec':spec})

        # write feat
        savename = os.path.join(outpath, w.rstrip(".wav"))
        head, tail = os.path.split(savename)
        if not os.path.exists(head):
                os.makedirs(head)
        np.savetxt(savename, spec)

if __name__ == '__main__':
    wavctl = './timit_phsegwav_test.ctl'
    wavpath = '../../feat_falign_extract/phsegwav_out'
    outpath = './phsegwav_feat'
    feat = 'spectrogram'

    compute_feat(wavctl, wavpath, outpath, feat)
