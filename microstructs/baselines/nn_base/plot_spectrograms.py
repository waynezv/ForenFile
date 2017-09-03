#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import visdom
import pdb

vis = visdom.Visdom()

path = "./phsegwav_feat/train"
phn_list = os.listdir(path)

rand_gen = np.random.RandomState(1234567)
for phn in phn_list:
    idx = rand_gen.randint(1, 50)
    img = np.loadtxt(os.path.join(path, phn, str(idx)))
    if img.ndim <= 1:
        continue
    H, W = img.shape
    img = img.reshape((1, H, W))
    vis.image(img)
    vis.text(phn + " " + str(idx))
