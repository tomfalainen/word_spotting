# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 08:53:26 2016

@author: tomas
"""
import sys
import json

import numpy as np

from util import MAP, MAP_qbs

if __name__ == '__main__':
    fold = int(sys.argv[1])
    tmp_file = str(sys.argv[2])
    ub = int(sys.argv[3])
    
    with open('data/washington/washington_preprocessed.json', 'r') as f:
        json_data = json.load(f)
    
    data = json_data['data']
    labels, texts, splits = [], [], []
    for datum in data:
        labels.append(datum['label'])
        texts.append(datum['text'])
        splits.append(datum['split'])
    
    X = np.load('tmp/%s_descriptors.npy' % tmp_file)
    
    if tmp_file.find('dct3') > 0:
        we = np.load('embeddings/washington_dct3_embeddings.npy')
    elif tmp_file.find('phoc') > 0:
        we = np.load('embeddings/washington_phoc_embeddings.npy')
        we = (we > 0).astype(np.float32)
    elif tmp_file.find('ngram') > 0:
        we = np.load('embeddings/washington_ngram_embeddings.npy')
    elif tmp_file.find('semantic') > 0:
        we = np.load('embeddings/washington_semantic_embeddings.npy')
    else:
        we = None
    
    #only keep train & val splits
    db = []
    itargets = []
    targets = []
    qtargets = []
    queries = []
    used = []
    if we != None:
        for i, (x, w, label, text, split) in enumerate(zip(X, we, labels, texts, splits)):
            if split == 'val' or split == 'train':
                db.append(x)
                itargets.append(label)
                targets.append(text)
                if label not in used:
                    queries.append(w)
                    qtargets.append(label)
                    used.append(label)
    else:
        for i, (x, label, text, split) in enumerate(zip(X, labels, texts, splits)):
            if split == 'val' or split == 'train':
                db.append(x)
                itargets.append(label)
                targets.append(text)
        
    db = np.array(db)
    itargets = np.array(itargets)
    targets = np.array(targets)
    
    if ub < 1:
        ub = db.shape[0] + 1
    
    #use entire db as query
    if we != None:
        mAP_qbs = MAP_qbs(queries[:ub], qtargets, db, itargets)
    else:
        mAP_qbs = -1
    mAP_qbe = MAP(db[:ub], itargets, db, itargets)
    
    jdata = {}
    jdata['MaP_qbe'] = mAP_qbe
    jdata['MaP_qbs'] = mAP_qbs
    #store results in a json file
    with open('tmp/' + tmp_file + '_ws_results.json', 'w') as f:
        json.dump(jdata, f)        
