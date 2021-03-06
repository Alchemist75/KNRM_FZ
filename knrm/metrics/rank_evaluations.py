# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import random
import numpy as np
import math

def conv2array(lis):
    output = np.zeros([len(lis)])
    for i in range(len(lis)):
	output[i] = lis[i]
    return output        
def calc_metric(metric_list,y_pred,y_label):
    res = {}
    arr_pred = conv2array(y_pred)
    arr_label = conv2array(y_label)
    for i in metric_list:
        if(i=="map"):
            res[i] = eval_map(arr_label,arr_pred)
        else:
            lis = i.strip().split('@')
            if (lis[0]=='precision'):
                res[i] = eval_precision(arr_label,arr_pred,k=int(lis[1]))
            else:
                if (lis[0]=='recall'):
                    res[i] = eval_recall(arr_label,arr_pred,k=int(lis[1]))
                else:
                    if (lis[0]=="ndcg"):
                        res[i] = eval_ndcg(arr_label,arr_pred,k=int(lis[1]))
    return res 
'''
class rank_eval():

    def __init__(self, rel_threshold=0.):
        self.rel_threshold = rel_threshold

    def zipped(self, y_true, y_pred):
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_true, y_pred)
        random.shuffle(c)
        return c

    def eval(self, y_true, y_pred, metrics=None, k = 20):
        if metrics is None:
            metrics = ['map', 'p@1', 'p@5', 'p@10', 'p@20', 'ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20']
        res = {}
        res['map'] = self.map(y_true, y_pred)
        all_ndcg = self.ndcg(y_true, y_pred, k=k)
        all_precision = self.precision(y_true, y_pred, k=k)
        res.update({'p@%d'%(i+1):all_precision[i] for i in range(k)})
        res.update({'ndcg@%d'%(i+1):all_ndcg[i] for i in range(k)})
        ret = {k:v for k,v in res.items() if k in metrics}
        return ret

    def map(self, y_true, y_pred):
        c = self.zipped(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        ipos = 0.
        s = 0.
        for i, (g,p) in enumerate(c):
            if g > self.rel_threshold:
                ipos += 1.
                s += ipos / ( 1. + i )
        if ipos == 0:
            return 0.
        else:
            return s / ipos

    def ndcg(self, y_true, y_pred, k = 20):
        s = 0.
        c = self.zipped(y_true, y_pred)
        c_g = sorted(c, key=lambda x:x[0], reverse=True)
        c_p = sorted(c, key=lambda x:x[1], reverse=True)
        #idcg = [0. for i in range(k)]
        idcg = np.zeros([k], dtype=np.float32)
        dcg = np.zeros([k], dtype=np.float32)
        #dcg = [0. for i in range(k)]
        for i, (g,p) in enumerate(c_g):
            if g > self.rel_threshold:
                idcg[i:] += (math.pow(2., g) - 1.) / math.log(2. + i)
            if i >= k:
                break
        for i, (g,p) in enumerate(c_p):
            if g > self.rel_threshold:
                dcg[i:] += (math.pow(2., g) - 1.) / math.log(2. + i)
            if i >= k:
                break
        for idx, v in enumerate(idcg):
            if v == 0.:
                dcg[idx] = 0.
            else:
                dcg[idx] /= v
        return dcg

    def precision(self, y_true, y_pred, k = 20):
        c = self.zipped(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        ipos = 0
        s = 0.
        precision = np.zeros([k], dtype=np.float32) #[0. for i in range(k)]
        for i, (g,p) in enumerate(c):
            if g > self.rel_threshold:
                precision[i:] += 1
            if i >= k:
                break
        precision = [v / (idx + 1) for idx, v in enumerate(precision)]
        return precision
'''

def eval_map(y_true, y_pred, rel_threshold=0):
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if (y_true.shape!=()):
        c = zip(y_true, y_pred)
    else:
        c = np.array([[y_true],[y_pred]])
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / ( j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s

def eval_ndcg(y_true, y_pred, k = 10, rel_threshold=0.):
    if k <= 0:
        return 0.
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if (y_true.shape!=()):
        c = zip(y_true, y_pred)
    else:
        c = np.array([[y_true],[y_pred]])
    #print(type(y_pred))
    #c = zip(y_true, y_pred)
    random.shuffle(c)
    c_g = sorted(c, key=lambda x:x[0], reverse=True)
    c_p = sorted(c, key=lambda x:x[1], reverse=True)
    idcg = 0.
    ndcg = 0.
    for i, (g,p) in enumerate(c_g):
        if i >= k:
            break
        if g > rel_threshold:
            idcg += g / math.log(2. + i) * math.log(2.)
    for i, (g,p) in enumerate(c_p):
        if i >= k:
            break
        if g > rel_threshold:
            ndcg += g / math.log(2. + i) * math.log(2.)
    if idcg == 0.:
        return 0.
    else:
        return ndcg / idcg

def eval_precision(y_true, y_pred, k = 10, rel_threshold=0.):
    if k <= 0:
        return 0.
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    #c = zip(y_true, y_pred)
    if (y_true.shape!=()):
        c = zip(y_true, y_pred)
    else:
        c = np.array([[y_true],[y_pred]])
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[1], reverse=True)
    ipos = 0
    precision = 0.
    for i, (g,p) in enumerate(c):
        if i >= k:
            break
        if g > rel_threshold:
            precision += 1
    precision /=  k
    return precision

def eval_mrr(y_true, y_pred, k = 10):
    s = 0.
    return s

# compute recall@k
# the input is all documents under a single query
def eval_recall(y_true,y_pred,k=10,rel_threshold=0.):
    if k <= 0:
        return 0.
    s = 0.
    #y_true = _to_list(np.squeeze(y_true).tolist()) # y_true: the ground truth scores for documents under a query
    #y_pred = _to_list(np.squeeze(y_pred).tolist()) # y_pred: the predicted scores for documents under a query
    y_true = np.squeeze(y_true) # y_true: the ground truth scores for documents under a query
    y_pred = np.squeeze(y_pred) # y_pred: the predicted scores for documents under a query
    pos_count = sum(i > rel_threshold for i in y_true) # total number of positive documents under this query
    #c = zip(y_true, y_pred)
    if (y_true.shape!=()):
        c = zip(y_true, y_pred)
    else:
        c = np.array([[y_true],[y_pred]])
    random.shuffle(c)
    c = sorted(c, key=lambda x: x[1], reverse=True)
    ipos = 0
    recall = 0.
    for i, (g, p) in enumerate(c):
        if i >= k:
            break
        if g > rel_threshold:
            recall += 1
    recall /= pos_count
    return recall

