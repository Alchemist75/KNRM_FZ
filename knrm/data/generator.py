# Copyright (c) 2017, Carnegie Mellon University. All rights reserved.
#
# Use of the K-NRM package is subject to the terms of the software license set
# forth in the LICENSE file included with this software, and also available at
# https://github.com/AdeDZY/K-NRM/blob/master/LICENSE

"""
ranking data generator
include:
    point wise generator, each time yield a:
        query-doc pair (x), y
        x is a dict: h['q'] = array of q (batch size * max q len)
                    h['d'] = array of d (batch size * max title len)
                    h['idf'] = None or array of query term idf (batch size * max q len)

    pair wise generator, each time yield a
        query-doc+, doc- pair (x), y
            the same x with pointwise's, with an additional key:
                h['d_aux'] = array of the second d (batch size * max title len)

    all terms are int ids


"""
import os
import random
import sys
sys.path.append("./")
from knrm.utils import *
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    List,
    Float,
    Bool,
)
import numpy as np
from numpy import genfromtxt
import logging
from io import StringIO
import sys
# reload(sys)
# sys.setdefaultencoding('UTF8')



class DataGenerator(Configurable):
    # title_in = Unicode('/bos/data1/sogou16/data/training/1m_title.pad_t50',
    #                    help='titles term id csv, must be padded').tag(config=True)
    max_q_len = Int(10, help='max q len').tag(config=True)
    max_d_len = Int(50, help='max document len').tag(config=True)
    query_per_iter = Int(2, help="queries").tag(config=True)
    batch_per_iter = Int(10, help="batch").tag(config=True)
    batch_size = Int(16, help="bs").tag(config=True)
    epoch_size = Int(100, help="es").tag(config=True)
    q_name = Unicode('q')
    d_name = Unicode('d')
    aux_d_name = Unicode('d_aux')
    idf_name = Unicode('idf')
    neg_sample = Int(1, help='negative sample').tag(config=True)
    load_litle_pool=Bool(False, help='load little pool at beginning').tag(conf=True)
    min_score_diff = Float(0, help='min score difference for click data generated pairs').tag(config=True)
    high_label = Float(2, help='highest label').tag(config=True)
    vocabulary_size = Int(2000000).tag(config=True)

    def setdict(self, word_dict, idf_dict):
        self.word_dict = word_dict
        self.idf_dict = idf_dict

    def __init__(self, pair_stream_dir, isPointwise, word_dict, idf_dict, **kwargs):
        super(DataGenerator, self).__init__(**kwargs)
        # self.m_title_pool = np.array(None)
        # if self.load_litle_pool and self.neg_sample:
        #     self._load_title_pool()
        self.setdict(word_dict, idf_dict)
        self.pair_stream_dir = pair_stream_dir
        self.qfile_list = self.get_file_list()
        if isPointwise:
            self.batch_list = self.pointwise_batch_generate(self.batch_size, with_idf=True)
        print "min_score_diff: ", self.min_score_diff
        print "generator's vocabulary size: ", self.vocabulary_size

    def get_file_list(self):
        qfile_list = []
        for dirpath, dirnames, filenames in os.walk(self.pair_stream_dir):
            for fn in filenames:
                if fn.endswith('.txt'):
                    qfile_list.append(os.path.join(dirpath, fn))
        return qfile_list

    # def _load_title_pool(self):
    #     if self.title_in:
    #         logging.info('start loading title pool [%s]', self.title_in)
    #         self.m_title_pool = genfromtxt(self.title_in, delimiter=',',  dtype=int,)
    #         logging.info('loaded [%d] title pool', self.m_title_pool.shape[0])

    def pointwise_batch_generate(self, batch_size, with_idf=False):
        qfile_list = self.qfile_list

        l_q = []
        l_qid = []
        l_uid = []
        l_d = []
        l_idf = []
        l_y = []
        batch_list = []

        for f in qfile_list:
            with open(f, "r") as pair_stream:
                for line in pair_stream:
                    qid, query, uid, doc, label = line.split('\t')
                    qid = qid.strip()
                    uid = uid.strip()
                    query = char2list(query.strip().split(), self.word_dict)
                    idf = getidf(query, self.idf_dict)
                    doc = char2list(doc.strip().split(), self.word_dict)
                    label = float(label)
                    l_q.append(query)
                    l_d.append(doc)
                    l_y.append(label)
                    l_qid.append(qid)
                    l_uid.append(uid)
                    l_idf.append(idf)

                    if len(l_q) >= batch_size:
                        Q = np.zeros([batch_size, self.max_q_len], dtype=np.int32)
                        for num, i in enumerate(l_q):
                            qlength = min(self.max_q_len, len(i))
                            Q[num][:qlength] = l_q[num][:qlength]
                        D = np.zeros([batch_size, self.max_d_len], dtype=np.int32)
                        for num, i in enumerate(l_d):
                            dlength = min(self.max_d_len, len(i))
                            D[num][:dlength] = l_d[num][:dlength]
                        if with_idf:
                            IDF = np.zeros([batch_size, self.max_q_len], dtype=float)
                            for num, i in enumerate(l_idf):
                                ilength = min(self.max_q_len, len(i))
                                IDF[num][:ilength] = l_idf[num][:ilength]
                        else:
                            IDF = np.ones(Q.shape, dtype=float)
			Y = np.zeros([batch_size,])
			for num,i in enumerate(l_y):
			    Y[num] = i
                        #Y = l_y
                        X = {self.q_name: Q, self.d_name: D, self.idf_name: IDF, "qid": l_qid, "uid": l_uid}
                        # yield X, Y
                        batch_list.append([X, Y])
                        l_q, l_d, l_y, l_idf, l_qid, l_uid = [], [], [], [], [], []

        if l_q:
            Q = np.zeros([len(l_q), self.max_q_len], dtype=int)
            for num, i in enumerate(l_q):
                qlength = min(self.max_q_len, len(i))
                Q[num][:qlength] = l_q[num][:qlength]
            D = np.zeros([len(l_q), self.max_d_len], dtype=int)
            for num, i in enumerate(l_d):
                dlength = min(self.max_d_len, len(i))
                D[num][:dlength] = l_d[num][:dlength]
            if with_idf:
                IDF = np.zeros([batch_size, self.max_q_len], dtype=float)
                for num, i in enumerate(l_idf):
                    ilength = min(self.max_q_len, len(i))
                    IDF[num][:ilength] = l_idf[num][:ilength]
            else:
                IDF = np.ones(Q.shape, dtype=float)
	    Y = np.zeros([batch_size,])
	    for num,i in enumerate(l_y):
	    	Y[num] = i
            #Y = l_y
            X = {self.q_name: Q, self.d_name: D, self.idf_name: IDF, "qid": l_qid, "uid": l_uid}
            # yield X, Y
            batch_list.append([X, Y])
        logging.info('point wise generator to an end')
        return batch_list

    def pointwise_generate(self):
        for X, Y in self.batch_list:
            yield X, Y

    def make_pair(self, query_per_iter):
        qfile_list = self.qfile_list

        uid_doc = {}
        qid_query = {}
        qid_label_uid = {}
        qid_uid_label = {}
        qid_idf = {}

        qfiles = random.sample(qfile_list, query_per_iter)
        for fn in qfiles:
            with open(fn) as file:
                for line in file:
                    qid, query, uid, doc, label = line.split('\t')
                    qid = qid.strip()
                    uid = uid.strip()
                    query = char2list(query.strip().split(), self.word_dict)
                    idf = getidf(query, self.idf_dict)
                    doc = char2list(doc.strip().split(), self.word_dict)
                    label = float(label)
                    qid_query[qid] = query
                    uid_doc[uid] = doc
                    qid_idf[qid] = idf
                    if qid not in qid_label_uid:
                        qid_label_uid[qid] = {}
                    if label not in qid_label_uid[qid]:
                        qid_label_uid[qid][label] = []
                    qid_label_uid[qid][label].append(uid)
                    if qid not in qid_uid_label:
                        qid_uid_label[qid] = {}
                    qid_uid_label[qid][uid] = label

        #make pair
        pair_list = []
        for qid in qid_label_uid:
            for hl in qid_label_uid[qid]:
                for ll in qid_label_uid[qid]:
                    if hl <= ll:
                        continue
                    if hl < self.high_label:
                        continue
                    if hl - ll <= self.min_score_diff:
                        continue
                    for dp in qid_label_uid[qid][hl]:
                        for dn in qid_label_uid[qid][ll]:
                            pair_list.append([qid, dp, dn])
        return qid_query, uid_doc, qid_label_uid, pair_list, qid_idf, qid_uid_label

    def pairwise_reader(self, with_idf=False):
        while True:
            qid_query, uid_doc, qid_label_uid, pair_list, qid_idf, qid_uid_label = self.make_pair(self.query_per_iter)
            print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            print 'Pair Instance Count:', len(pair_list)
            for _i in range(self.batch_per_iter):
                sample_pair_list = random.sample(pair_list, self.batch_size)
                Xq = np.zeros((self.batch_size, self.max_q_len), dtype=np.int32)
                Xidf = np.zeros((self.batch_size, self.max_q_len), dtype=np.float)
                Xd = np.zeros((self.batch_size , self.max_d_len), dtype=np.int32)
                Xd_aux = np.zeros((self.batch_size , self.max_d_len), dtype=np.int32)
                #label not used
                Y = np.zeros((self.batch_size,2), dtype=np.float)

                for i in range(self.batch_size):
                    qid, dp_id, dn_id = sample_pair_list[i]
                    query = qid_query[qid]
                    dp = uid_doc[dp_id]
                    dn = uid_doc[dn_id]
                    idf = qid_idf[qid]
                    query_len = min(self.max_q_len, len(query))
                    dp_len = min(self.max_d_len, len(dp))
                    dn_len = min(self.max_d_len, len(dn))

                    Xq[i, :query_len] = query[:query_len]

                    Xd[i, :dp_len] = dp[:dp_len]
                    Xd_aux[i, :dn_len] = dn[:dn_len]
                    Xidf[i,:query_len] = idf[:query_len]
                    Y[i,0] = qid_uid_label[qid][dp_id]
                    Y[i,1] = qid_uid_label[qid][dn_id]
                if with_idf:
                    IDF = np.array(Xidf, dtype=float)
                else:
                    IDF = np.ones(Xidf.shape,dtype=float)
                X = {self.q_name: Xq, self.d_name: Xd, self.idf_name: IDF, self.aux_d_name: Xd_aux}
                yield X, Y
