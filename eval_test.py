# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:10:44 2016

@author: tomas
"""
import json
import numpy as np

import misc.util as util
embeddings = ['triplet', 'semantic', 'ngram', 'phoc', 'dct3']

with open("data/washington/washington_preprocessed.json") as f:
    jdata = json.load(f)

folds = 1
for embedding in embeddings:
    qbe, qbs = [], []
    for fold in range(1, folds + 1):
        descr = np.load("descriptors/washington_%s_fold_%d_descriptors.npy" % (embedding, fold))
        
        db = []
        itar = []
        texts = []
        for datum, desc in zip(jdata['data'], descr):
            if datum['split'] == 'test':
                db.append(desc)
                itar.append(datum['label'])
                texts.append(datum['text'])
        
        itar = np.array(itar)
        db = np.array(db)
        
        #Load word vectors
        if embedding == 'phoc':
            we = np.load('embeddings/washington_phoc_embeddings.npy')
        elif embedding == 'dct3':
            we = np.load("embeddings/washington_dct3_embeddings.npy")
        elif embedding == 'semantic':
            we = np.load("embeddings/washington_semantic_embeddings.npy")
        elif embedding == 'ngram':
            we = np.load("embeddings/washington_ngram_embeddings.npy")
        else:
    	      we = None

        if we != None:        
            queries = []
            qtargets = []
            qtrans = []        
            wtoe = {}
            for i, datum in enumerate(jdata['data']):
                if datum['split'] == 'test':
                    if datum['text'] not in wtoe:
                        wtoe[datum['text']] = we[i]
                        queries.append(we[i])
                        qtrans.append(datum['text'])
                        qtargets.append(datum['label'])
         
            queries = np.array(queries)        
            qtargets = np.array(qtargets)        
            assert(len(qtargets) == np.unique(qtargets).shape[0])
        
        if embedding == 'triplet':
            mAP_qbs = -1
        else:
            mAP_qbs = util.MAP_qbs(queries, qtargets, db, itar)
            
        mAP_qbe = util.MAP(db, itar, db, itar)

        qbe.append(mAP_qbe)
        qbs.append(mAP_qbs)
   
    print embedding, "Washington QbE ", np.mean(qbe) * 100, ", QbS:", np.mean(qbs) * 100
