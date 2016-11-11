# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:41:28 2016

@author: tomas
"""

import string
import numpy as np
import scipy.fftpack as sf

import db_utils as dbu

alphabet = string.ascii_lowercase + string.digits
dct_resolution = 3 #Resolution of dct to keep

def plain_dct(s, resolution, alphabet):
    im = np.zeros([len(alphabet),len(s)], 'single')
    F = np.zeros([len(alphabet),len(s)], 'single')
    for jj in range(0,len(s)):
        c = s[jj]
        im[string.find(alphabet, c), jj] = 1.0

    for ii in range(0,len(alphabet)):
        F[ii,:] = sf.dct(im[ii,:])

    A = F[:,0:resolution]
    B = np.zeros([len(alphabet),max(0,resolution-len(s))])

    return np.hstack((A,B))

fold = 1
data = dbu.load_washington(fold, "../data/washington")

embeddings = []
for datum in data:
    embedding = plain_dct(datum['text'], dct_resolution, alphabet)
    embedding = embedding.reshape(-1)
    embeddings.append(embedding)

embeddings = np.array(embeddings)

np.save('../embeddings/washington_dct%d_embeddings.npy' % dct_resolution, embeddings)
    
