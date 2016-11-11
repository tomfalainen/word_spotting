# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:35:01 2016

@author: tomas
"""
import numpy as np
from scipy.spatial import distance as di

def average_precision(ils, t):
    '''
    Computes the average precision
    
    Thanks for a fast mAP implementation!
    https://github.com/ssudholt/phocnet/blob/master/src/phocnet/evaluation/retrieval.py
    '''
    
    ret_vec_relevance =  ils == t
    ret_vec_cumsum = np.cumsum(ret_vec_relevance, dtype=float)
    ret_vec_range = np.arange(1, ret_vec_relevance.size + 1)
    ret_vec_precision = ret_vec_cumsum / ret_vec_range
    
    n_relevance = ret_vec_relevance.sum()

    if n_relevance > 0:
        ret_vec_ap = (ret_vec_precision * ret_vec_relevance).sum() / n_relevance
    else:
        ret_vec_ap = 0.0
    
    return ret_vec_ap
    
def MAP(queries, qtargets, db, itargets, metric='cosine'):
    APs = []
    for q, query in enumerate(queries):
        t = qtargets[q]     #Get the label for the query
        count = np.sum(itargets == t)    #Count the number of relevant retrievals in the database        
        if count == 1:
            continue

        dists = np.squeeze(di.cdist(query.reshape(1, query.shape[0]), db, metric=metric))   
        I = np.argsort(dists)
        I = I[1:]           #Don't count the query, distance is always zero to query
        ils = itargets[I]   #Sort results after distance to query image
        ap = average_precision(ils, t)
        APs.append(ap) 

    return np.mean(APs)
    
def MAP_qbs(queries, qtargets, db, itargets, metric='cosine'):
    APs = []
    for q, query in enumerate(queries):
        t = qtargets[q]     #Get the label for the query
        dists = np.squeeze(di.cdist(query.reshape(1, query.shape[0]), db, metric=metric))   
        I = np.argsort(dists)
        ils = itargets[I]   #Sort results after distance to query image
        ap = average_precision(ils, t)
        APs.append(ap) 
    return np.mean(APs)#, qs    
        