# Copyright (c) 2017, Carnegie Mellon University. All rights reserved.
#
# Use of the K-NRM package is subject to the terms of the software license set
# forth in the LICENSE file included with this software, and also available at
# https://github.com/AdeDZY/K-NRM/blob/master/LICENSE

"""
basic utility functions
    set_basic_log,
    load_trec_ranking,
    load_q_info_from_trec,
    load_doc_info_from_trec,
    load_q_info,
    load_doc_info,
    dump_trec_ranking,
"""
import numpy as np
import sys
import logging
import logging.handlers
import json
import time
#from deeplearning4ir.utils.base_conf import *
import subprocess
from traitlets.config import PyFileConfigLoader



def read_embedding(filename):
    embed = {}
    word_dict = {}
    idf_dict = {}
    cnt = -1
    # max_cnt = max(word_ids.keys())
    for line in open(filename):
        cnt += 1
        if cnt == 0:
            continue
        attr = line.strip().split()
        if len(attr) != 52:
            print 'embed file error!'
            exit()
        term = attr[0].strip().lower()
        word_dict[term] = cnt
        idf_dict[cnt] = float(attr[1])
        embed[cnt] = map(float, attr[2:])
        # if cnt == 1000:
        #     break
    embed_array = np.random.normal(size=[cnt+1,len(embed[1])])
    # for i in range(1, cnt+1):
    for i in embed:
        embed_array[i] = np.array(embed[i]) 

    print '[%s]'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '[%s]\n\tEmbedding size: %d' % (filename, len(embed))
    return embed_array, cnt+1, len(embed[1]), word_dict, idf_dict

def char2list(text,word_dict):
    new_text = []
    for term in text:
        term = term.strip().lower()
        if len(term) < 1:
            new_text.append(0)
            continue
        if term in word_dict:
            new_text.append(word_dict[term])
            if (word_dict[term]<-10):
                print "WRONG CHAR!! \n", term, word_dict[term]
                exit()
        else:
            new_text.append(0)
    assert len(new_text) == len(text)
    return new_text

def getidf(text,idf_dict):
    idf = []
    for term in text:
        if term in idf_dict:
            idf.append(idf_dict[term])
            if (idf_dict[term]<-10):
                print "WRONG IDF!! \n", term, idf_dict[term]
                exit()
        else:
            idf.append(0)

    assert len(idf) == len(text)
    return idf
'''

def set_basic_log(log_level=logging.INFO):
    """
    set basic logs
    :param log_level:
    :return:
    """
    root = logging.getLogger()
    root.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)


def get_user_job(user='cx'):
    out_str = subprocess.check_output(['condor_q', user])
    l_job_id = [line.split()[0] for line in out_str.splitlines() if user in line]
    return l_job_id


def qsub_job(l_cmd):
    out_str = subprocess.check_output(['qsub'] + l_cmd)
    l_job_id = [line.strip('.').split()[-1]
                for line in out_str.splitlines() if 'submitted to cluster' in line]
    logging.info('submit %s to %s', json.dumps(l_cmd), l_job_id[0])
    return l_job_id[0]


def load_svm_feature(in_name):
    """
    load svm format data
    :param in_name: svm in
    :return: {qid, h_feature, score, and comment}
    """

    l_svm_data = []

    for line in open(in_name):
        line = line.strip()
        cols = line.split('#')
        data = cols[0]
        comment = ""
        if len(cols) > 1:
            comment = '#'.join(cols[1:])

        cols = data.split()
        qid = cols[1].replace('qid:', '')
        score = float(cols[0])

        feature_cols = cols[2:]
        l_feature = [item.split(':') for item in feature_cols]
        l_feature = [(int(item[0]), float(item[1])) for item in l_feature]
        h_feature = dict(l_feature)
        l_svm_data.append({
            'qid': qid,
            'score': score,
            'feature': h_feature,
            'comment': comment
        })
    logging.info('load [%d] svm data line from [%s]', len(l_svm_data), in_name)
    return l_svm_data


def dump_svm_feature(l_svm_data, out_name):
    out = open(out_name, 'w')
    l_svm_data.sort(key=lambda item: int(item['qid'])) # sort
    for svm_data in l_svm_data:
        print >>out, _dumps_svm_line(svm_data)
    out.close()
    logging.info('dump [%d] svm line to [%s]', len(l_svm_data), out_name)
    return


def _dumps_svm_line(svm_data):
    res = '%f qid:%s' % (svm_data['score'], svm_data['qid'])
    l_feature = svm_data['feature'].items()
    l_feature.sort(key=lambda item: int(item[0]))
    l_feature_str = ['%d:%.6f' % (item[0], item[1]) for item in l_feature]
    res += ' ' + ' '.join(l_feature_str)
    res += ' # ' + svm_data['comment']
    return res


def load_trec_labels_dict(in_name):
    """
    input: trec format qrel
    :param in_name:  qrel
    :return: h_qrel = {qid:{doc:score} }
    """
    h_qrel = {}
    l_lines = open(in_name).read().splitlines()
    for line in l_lines:
        cols = line.split()
        qid = cols[0].strip()
        docno = cols[2].strip()
        label = int(cols[3])
        if qid not in h_qrel:
            h_qrel[qid] = {}
        h_qrel[qid][docno] = label

    return h_qrel


def load_trec_labels(in_name):
    h_qrel = load_trec_labels_dict(in_name)
    l_qrel = h_qrel.items()
    l_qrel.sort(key=lambda item: int(item[0]))
    return l_qrel


def dump_trec_labels(l_qrel, out_name):
    out = open(out_name, 'w')
    l_qrel.sort(key=lambda item: int(item[0]))
    for qid, h_doc_score in l_qrel:
        for docno, label in h_doc_score.items():
            print >> out, qid + ' 0 ' + docno + ' ' + str(label)
    out.close()
    logging.debug('[%d] q\'s relevance dumped to [%s]', len(l_qrel), out_name)
    return


def load_trec_ranking(in_name):
    ll_qid_ranked_doc = []

    this_qid = None

    for line in open(in_name):
        cols = line.strip().split()
        qid = cols[0]
        docno = cols[2]
        score = float(cols[4])

        if qid != this_qid:
            ll_qid_ranked_doc.append([qid, []])
            this_qid = qid
        ll_qid_ranked_doc[-1][-1].append([docno, score])
    logging.info('trec ranking loaded [%s]', in_name)
    return ll_qid_ranked_doc


def load_py_config(in_name):
    reader = PyFileConfigLoader(in_name)
    reader.load_config()
    logging.info('load from [%s] conf: %s', in_name, reader.config)
    return reader.config


def dump_trec_ranking(ll_qid_ranked_doc, out_name, ranking_name='na'):
    out = open(out_name, 'w')
    ll_mid = list(ll_qid_ranked_doc)
    ll_mid.sort(key=lambda item: int(item[0]))
    for l_qid_ranking in ll_mid:
        qid, ranking = l_qid_ranking
        for p, (doc, score) in enumerate(ranking):
            print >> out, '%s\tQ0\t%s\t%d\t%f # %s' % (
                qid,
                doc,
                p + 1,
                score,
                ranking_name
            )
    out.close()
    logging.info('trec ranking dumped to [%s]', out_name)


def dump_trec_out_from_ranking_score(l_qid, l_docno, l_score, out_name, method_name='na'):
    l_data = zip(l_qid, zip(l_docno, l_score))
    l_data.sort(key=lambda item: (int(item[0]), -item[1][1]))

    out = open(out_name, 'w')
    rank_p = 1
    this_qid = None
    for qid, (docno, score) in l_data:
        if this_qid is None:
            this_qid = qid

        if qid != this_qid:
            rank_p = 1
            this_qid = qid
        print >> out, '%s Q0 %s %d %f # %s' %(
            qid, docno, rank_p, score,
            method_name,
        )
        rank_p += 1

    out.close()
    logging.debug('ranking result dumped to [%s]', out_name)


def load_q_info_from_trec(trec_in_name):
    """
    qid: dict
    :param trec_in_name: trec rankings file with additional stuff after #
    :return:
    """
    h_q_info = {}
    for line in open(trec_in_name):
        line = line.strip()
        qid = line.split()[0]
        query = json.loads('#'.join(line.split('#')[1:]))['query']
        h_q_info[qid] = {'query': query}
    return h_q_info


def load_doc_info_from_trec(trec_in_name):
    h_doc_info = {}
    for line in open(trec_in_name):
        line = line.strip()
        docno = line.split()[2]
        h_doc_info[docno] = json.loads('#'.join(line.split('#')[1:]))['doc']
    return h_doc_info


# def load_target_info_from_trec(trec_in_name, target_key='q', target_id='qid'):
#     h_key_info = {}
#     lines = open(trec_in_name).read().splitlines()
#
#     l_h_data = [json.loads(line.split('#')[-1]) for line in lines]
#     l_qid = [line.split()[0] for line in lines]
#     l_docno = [line.split()[2] for line in lines]
#     for h_data in l_h_data:
#         if target_key in h_data:
#             if target_id in h_data[target_key]:
#                 mid = h_data[target_key][target_id]
#                 h_key_info[mid] = h_data[target_key][target_id]
#     logging.info('[%d] info loaded from trec file [%s]', len(h_key_info), trec_in_name)
#
#     return h_key_info


def load_target_info(in_name, target_id='qid'):
    h_key_info = {}

    for line in open(in_name):
        h_data = json.loads(line)
        if target_id in h_data:
            h_key_info[target_id].update(h_data[target_id])
    return h_key_info


def load_q_info(in_name):
    return load_target_info(in_name, 'qid')


def load_doc_info(in_name):
    return load_target_info(in_name, 'docno')


def text_to_lm(text):
    terms = text.split()
    h_t = {}
    for t in terms:
        if t not in h_t:
            h_t[t] = 1
        else:
            h_t[t] += 1
    return h_t


def eval_trec(in_name, out_name, qrel_in=QREL_PATH):
    l_cmd = ['perl', GDEVAL_PATH, qrel_in, in_name]
    res_str = subprocess.check_output(l_cmd)
    print >> open(out_name, 'w'), res_str.strip()
    mean_line = [line for line in res_str.splitlines() if 'amean' in line][0]
    logging.info('[%s] performance: %s', in_name, mean_line)
    ndcg, err = mean_line.split(',')[-2:]
    ndcg = float(ndcg)
    err = float(err)
    return ndcg, err

def load_gdeval_res(in_name):
    return seg_gdeval_out(open(in_name).read())


def seg_gdeval_out(eva_str, with_mean=True):
    l_qid_eva = []
    mean_ndcg = 0
    mean_err = 0
    for line_cnt, line in enumerate(eva_str.splitlines()):
        if line_cnt == 0:
            continue
        qid, ndcg, err = line.split(',')[-3:]
        ndcg = float(ndcg)
        err = float(err)
        if qid == 'amean':
            mean_ndcg = ndcg
            mean_err = err
        else:
            l_qid_eva.append([qid, (ndcg, err)])
    # logging.info('get eval res %s, mean %f,%f', json.dumps(l_qid_eva), mean_ndcg, mean_err)
    l_qid_eva.sort(key=lambda item: int(item[0]))
    if with_mean:
        return l_qid_eva, mean_ndcg, mean_err
    else:
        return l_qid_eva
'''