# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:57:40 2016

@author: tomas
"""
import os
import copy
import json

import numpy as np
from skimage.io import imsave
from skimage.io import imread
from skimage.util import img_as_ubyte
import skimage.morphology as mor
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import skimage.transform as tf

def close_crop(img, tparams):
    t_img = img < threshold_otsu(img)
    nz = t_img.nonzero()
    pad = np.random.randint(low = tparams['hpad'][0], high = tparams['hpad'][1], size=2)    
    vpad = np.random.randint(low = tparams['vpad'][0], high = tparams['vpad'][1], size=2)    
    b = [max(nz[1].min() - pad[0], 0), max(nz[0].min() - vpad[0], 0), 
         min(nz[1].max() + pad[1], img.shape[1]), min(nz[0].max() + vpad[1], img.shape[0])]
    return img[b[1]:b[3], b[0]:b[2]]
    
def affine(img, tparams):
    phi = (np.random.uniform(tparams['shear'][0], tparams['shear'][1])/180) * np.pi
    theta = (np.random.uniform(tparams['rotate'][0], tparams['rotate'][1])/180) * np.pi
    t = tf.AffineTransform(shear=phi, rotation=theta, translation=(-25, -50))
    tmp = tf.warp(img, t, order=tparams['order'], mode='edge', output_shape=(img.shape[0] + 100, img.shape[1] + 100))
    return tmp

    
def morph(img, tparams):
    ops = [mor.grey.erosion, mor.grey.dilation]
    t = ops[np.random.randint(2)] 
    if t == 0:    
        selem = mor.square(np.random.randint(1, tparams['selem_size'][0]))
    else:
        selem = mor.square(np.random.randint(1, tparams['selem_size'][1]))
    return t(img, selem)    
    
def augment(datum, tparams):
    img = imread(datum['file'])
    if img.ndim == 3:
        img = img_as_ubyte(rgb2gray(img))

    tmp = np.ones((img.shape[0] + 8, img.shape[1] + 8), dtype = img.dtype) * 250
    tmp[4:-4, 4:-4] = img
    img = tmp
    img = affine(img, tparams)
    img = close_crop(img, tparams)        
    img = morph(img, tparams)
    img = img_as_ubyte(img)
    return img


def augment_data(data, out_directory, tparams=None, M=500000):
    """
    Augments a dataset using basic affine and morphological transformations
    data: Data to augment
    out_directory: Directory to save the augmented training examples
    tparams: Dictionary of parameters for the augmentation transformations
    M: size of augmented dataset
    """
    output_json = os.path.join(out_directory, 'data.json')
    if os.path.exists(output_json): #if augmentation already done, load
        with open(output_json) as f:
            new_data = json.load(f)
    
    else: #else, augment
        if tparams == None:
            tparams = {}
            tparams['shear'] = (-5, 30)
            tparams['rotate'] = (-5, 5)
            tparams['hpad'] = (3, 15)
            tparams['vpad'] = (3, 15)
            tparams['order'] = 1            #bilinear
            tparams['selem_size'] = (3, 4)  #max size for square selem for erosion, dilation
    
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        
        train_data = []
        texts = []
        for datum in data:
            if datum['split'] == 'train' or datum['split'] == 'val':
                texts.append(datum['text'])
                train_data.append(copy.deepcopy(datum))
                
        vocab = sorted(list(set(texts)))
        wtoi = {w:i for i,w in enumerate(vocab)} # inverse table
        data_by_class = [[] for v in vocab]
        for datum, text in zip(train_data, texts):
            data_by_class[wtoi[text]].append(datum)
        
        counts = [len(l) for l in data_by_class]
        mc = max(counts)
        
        print "starting stage 1"
        new_data = []
        k = 0
        #Stage 1: get an equal amount of each class
        for i in range(len(vocab)):
            n = mc - counts[i]
            if n > 0:
                for j in range(n):
                    class_data = data_by_class[i]
                    ind = np.random.randint(len(class_data))
                    datum = copy.deepcopy(class_data[ind])
                    img = augment(datum, tparams)
                    f = os.path.join(out_directory, datum['file'].split('/')[-1][:-4] + '_%d_%s.png' % (k, datum['split']))
                    imsave(f, img)
                    datum['file'] = f
                    new_data.append(datum)
                    k += 1
        
        print "stage 1 done"
        
        #Append existing data to augmented data to get an equal distribution for each word
        new_data += train_data
        
        #Stage 2: Continue sampling new training data until we reach the desired amount
        N = len(new_data)
        lv = len(vocab)
        
        left = M - N
        if left > 0:
            for j in range(left):
                i = j % lv
                class_data = data_by_class[i]
                ind = np.random.randint(len(class_data))
                datum = copy.deepcopy(class_data[ind])
                img = augment(datum, tparams)
                f = os.path.join(out_directory, datum['file'].split('/')[-1][:-4] + '_%d_%s.png' % (k + j, datum['split']))
                imsave(f, img)
                datum['file'] = f
                new_data.append(datum)
                
        print "stage 2 done"
        
        with open(output_json, 'w') as f:
            json.dump(new_data, f)
    
    return new_data
