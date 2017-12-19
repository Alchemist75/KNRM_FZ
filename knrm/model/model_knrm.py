# Copyright (c) 2017, Carnegie Mellon University. All rights reserved.
#
# Use of the K-NRM package is subject to the terms of the software license set
# forth in the LICENSE file included with this software, and also available at
# https://github.com/AdeDZY/K-NRM/blob/master/LICENSE
import argparse
import sys
import time
import numpy as np
import tensorflow as tf
sys.path.append("./")
from knrm.utils import char2list, getidf, read_embedding
from knrm.data import DataGenerator
from knrm.model import BaseNN
from knrm.metrics import calc_metric
from traitlets import Bool, Float, Int, Unicode
from traitlets.config import Configurable
from traitlets.config.loader import PyFileConfigLoader
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# reload(sys)
# sys.setdefaultencoding('UTF8')

class Knrm(BaseNN):
    neg_sample = 1

    emb_in = Unicode('None', help="initial embedding. Terms should be hashed to ids.").tag(config=True)
    train_in = Unicode('None', help="initial train.").tag(config=True)
    test_in  = Unicode('None', help="initial test.").tag(config=True)
    valid_in_list = Unicode('None', help="initial valid.").tag(config=True)
    checkpoint_dir = Unicode('None', help="checkpoint saver's path.").tag(config=True)
    save_frequency = Int(100, help="save frequency").tag(config=True)
    print_frequency = Int(2, help="print score frequency").tag(config=True)
    metrics = Unicode('None', help="Metrics.").tag(config=True)
    lamb = Float(0.5, help="guassian_sigma = lamb * bin_size").tag(config=True)
    learning_rate = Float(0.001, help="learning rate, default is 0.001").tag(config=True)
    epsilon = Float(0.00001, help="Epsilon for Adam").tag(config=True)


    def __init__(self, **kwargs):
        super(Knrm, self).__init__(**kwargs)

        self.mus = Knrm.kernal_mus(self.n_bins, use_exact=True)
        self.sigmas = Knrm.kernel_sigmas(self.n_bins, self.lamb, use_exact=True)
        print "kernel sigma values: ", self.sigmas
        print '"num_batch": %d'%self.num_batch

        print "trying to load initial embeddings from:  ", self.emb_in
        assert self.emb_in != 'None'
        # self.embed, self.vocabulary_size, self.embedding_size, self.word_dict, self.idf_dict = read_embedding(self.emb_in)
        self.embeddings = tf.Variable(tf.constant(self.embed, dtype='float32', shape=[self.vocabulary_size, self.embedding_size]))
        # print "Initialized embeddings with {0}".format(self.emb_in)
        print "Initialized embeddings!"
        # Model parameters for feedfoward rank NN
        self.W1 = Knrm.weight_variable([self.n_bins, 1])
        self.b1 = tf.Variable(tf.zeros([1]))
        # self.data_generator.setdict(self.word_dict,self.idf_dict)
        # for i in range(len(self.val_data_generator)):
        #     self.val_data_generator[i].setdict(self.word_dict,self.idf_dict)
        # self.val_data_generator.setdict(self.word_dict,self.idf_dict)
        # self.test_data_generator.setdict(self.word_dict,self.idf_dict)
        assert self.metrics != 'None'
        self.metriclist = self.metrics.strip().split(';')

    def model(self, inputs_q, inputs_d, mask, q_weights, mu, sigma):
        """
        The pointwise model graph
        :param inputs_q: input queries. [nbatch, qlen, emb_dim]
        :param inputs_d: input documents. [nbatch, dlen, emb_dim]
        :param mask: a binary mask. [nbatch, qlen, dlen]
        :param q_weights: query term weigths. Set to binary in the paper.
        :param mu: kernel mu values.
        :param sigma: kernel sigma values.
        :return: return the predicted score for each <query, document> in the batch
        """
        # look up embeddings for each term. [nbatch, qlen, emb_dim]

        q_embed = tf.nn.embedding_lookup(self.embeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(self.embeddings, inputs_d, name='demb')

        ## Uingram Model
        # normalize and compute similarity matrix
        norm_q = tf.sqrt(tf.reduce_sum(tf.square(q_embed), 2, keep_dims=True))
        normalized_q_embed = q_embed / norm_q
        norm_d = tf.sqrt(tf.reduce_sum(tf.square(d_embed), 2, keep_dims=True))
        normalized_d_embed = d_embed / norm_d
        tmp = tf.transpose(normalized_d_embed, perm=[0, 2, 1])

        # similarity matrix [n_batch, qlen, dlen]
        sim = tf.batch_matmul(normalized_q_embed, tmp, name='similarity_matrix')

        # compute gaussian kernel
        rs_sim = tf.reshape(sim, [self.batch_size, self.max_q_len, self.max_d_len, 1])

        # compute Gaussian scores of each kernel
        tmp = tf.exp(-tf.square(tf.sub(rs_sim, mu)) / (tf.mul(tf.square(sigma), 2)))

        # mask those non-existing words.
        tmp = tmp * mask

        feats = []  # store the soft-TF features from each field.
        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, [2])
        kde = tf.log(tf.maximum(kde, 1e-10)) * 0.01  # 0.01 used to scale down the data.
        # [batch_size, qlen, n_bins]

        # aggregated query terms
        # q_weights = [1, 1, 0, 0...]. Works as a query word mask.
        # Support query-term weigting if set to continous values (e.g. IDF).
        aggregated_kde = tf.reduce_sum(kde * q_weights, [1])  # [batch, n_bins]

        feats.append(aggregated_kde) # [[batch, nbins]]
        feats_tmp = tf.concat(1, feats)  # [batch, n_bins]
        #print "batch feature shape:", feats_tmp.get_shape()

        # Reshape. (maybe not necessary...)
        feats_flat = tf.reshape(feats_tmp, [-1, self.n_bins])
        #print "flat feature shape:", feats_flat.get_shape()

        # Learning-To-Rank layer. o is the final matching score.
        o = tf.tanh(tf.matmul(feats_flat, self.W1) + self.b1)

        # data parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            #print(shape)
            #print(len(shape))
            variable_parametes = 1
            for dim in shape:
                #print(dim)
                variable_parametes *= dim.value
            #print(variable_parametes)
            total_parameters += variable_parametes
        print "total number of parameters:", total_parameters

        # return some mid result and final matching score.
        return (sim, feats_flat), o

    def train(self, train_size, load_model=False):

        # PLACEHOLDERS
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.

        train_pair_file_path = self.train_in
        val_pair_file_path_list = self.valid_in_list.strip().split(";")
        test_pair_file_path = self.test_in
        # nodes to hold mu sigma
        input_mu = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_mu')
        input_sigma = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_sigma')

        # nodes to hold query and qterm idf. padding terms will have idf=0
        train_inputs_q = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_q_len], name='train_inputs_q')
        train_input_q_weights = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len], name='idf')

        # nodes to hold training data, postive samples
        train_inputs_pos_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len],
                                            name='train_inputs_pos_d')

        # nodes to hold negative samples
        train_inputs_neg_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len])

        # mask padding terms
        # assume all termid >= 1
        # padding with 0
        input_train_mask_pos = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])
        input_train_mask_neg = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])

        input_labels = tf.placeholder(tf.float32, shape=[self.batch_size,2])

        # reshape place holders
        mu = tf.reshape(input_mu, shape=[1, 1, self.n_bins])
        sigma = tf.reshape(input_sigma, shape=[1, 1, self.n_bins])
        rs_train_mask_pos = tf.reshape(input_train_mask_pos, [self.batch_size, self.max_q_len, self.max_d_len, 1])
        rs_train_mask_neg = tf.reshape(input_train_mask_neg, [self.batch_size, self.max_q_len, self.max_d_len, 1])
        rs_q_weights = tf.reshape(train_input_q_weights, shape=[self.batch_size, self.max_q_len, 1])

        # training graph
        mid_res_pos, o_pos = self.model(train_inputs_q, train_inputs_pos_d, rs_train_mask_pos, rs_q_weights, mu, sigma)
        mid_res_neg, o_neg = self.model(train_inputs_q, train_inputs_neg_d, rs_train_mask_neg, rs_q_weights, mu, sigma)
        
        o_all = tf.concat(1,[o_pos,o_neg])
        #print o_all
        o_all = tf.nn.softmax(o_all)
        softmaxed_label = tf.nn.softmax(input_labels)
        #print o_all
        #o_pos = o_all[:,0]
        #o_neg = o_all[:,1]
        #print o_pos, o_neg
        #loss = tf.reduce_mean(tf.maximum(0.0, 1 - o_pos + o_neg))       
        
        bottom = -tf.reduce_sum(softmaxed_label*tf.log(tf.clip_by_value(softmaxed_label,0.0001,1-0.0001)),axis=-1)
        loss = tf.reduce_mean(-tf.reduce_sum(softmaxed_label*tf.log(tf.clip_by_value(o_all,0.0001,1-0.0001)),axis=-1) - bottom)
        

        
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon).minimize(loss)

        # Create a local session to run the training.
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        with tf.Session(config=config_tf) as sess:

            saver = tf.train.Saver()
            # start_time = time.time()

            # Run all the initializers to prepare the trainable parameters.

            if not load_model:
                print "Initializing a new model..."
                tf.initialize_all_variables().run()
                print('New model initialized!')
            else:
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print "model loaded!"
                else:
                    print "no data found"
                    exit(-1)

            print "Initializing a new model..."
            tf.initialize_all_variables().run()
            print('New model initialized!')

            # Loop through training steps.
            # step = 0
            for epoch in range(int(self.max_epochs)):
                # print "EPOCH: "+str(epoch)

                #pair_stream = open(train_pair_file_path)
                ##########  TRAIN  ###########
                self.train_in_train(train_inputs_q, train_inputs_pos_d, train_inputs_neg_d, train_input_q_weights,
                                    input_mu,  input_sigma, input_train_mask_pos, input_train_mask_neg, input_labels, sess,
                                    optimizer, loss, epoch)

                ##########  VALIDATION  ###########
                if (epoch % self.print_frequency == 0):
                    output = open('../MatchZoo_zyk/output/%s/%s_%s_output_%s.txt' % ("K-NRM", "K-NRM", 'val', str(epoch+1)), 'w')
                else:
                    output = None
                self.valid_in_train(val_pair_file_path_list, train_inputs_q, train_inputs_pos_d,
                                    train_inputs_neg_d, train_input_q_weights, input_mu, input_sigma,
                                    input_train_mask_pos, input_train_mask_neg, sess, o_pos,  epoch, loss, output)
                if (epoch % self.print_frequency == 0):
                    output.close()

                ##########  TEST  ###########
                if (epoch % self.print_frequency == 0):
                    output = open('../MatchZoo_zyk/output/%s/%s_%s_output_%s.txt' % ("K-NRM", "K-NRM", 'test', str(epoch+1)), 'w')
                else:
                    output = None
                self.test_in_train(train_inputs_q, train_inputs_pos_d, train_inputs_neg_d,
                                   train_input_q_weights, input_mu, input_sigma, input_train_mask_pos,
                                   input_train_mask_neg, sess, o_pos, epoch, output)
                if (epoch % self.print_frequency == 0):
                    output.close()

                # save data
                if (epoch % self.save_frequency == 0):
                    saver.save(sess, self.checkpoint_dir + '/data.ckpt')
                # END epoch
                print ''

            # end training
            saver.save(sess, self.checkpoint_dir + '/data.ckpt')

    def train_in_train(self, train_inputs_q, train_inputs_pos_d, train_inputs_neg_d, train_input_q_weights,
                       input_mu,  input_sigma, input_train_mask_pos, input_train_mask_neg, input_labels, sess,
                       optimizer, loss, epoch):
        batch_step = 0
        for BATCH in self.data_generator.pairwise_reader(with_idf=True):
            batch_step += 1
            # step += 1
            X, Y = BATCH
            M_pos = self.gen_mask(X[u'q'], X[u'd'])
            M_neg = self.gen_mask(X[u'q'], X[u'd_aux'])
            train_feed_dict = {train_inputs_q: self.re_pad(X[u'q'], self.batch_size),
                               train_inputs_pos_d: self.re_pad(X[u'd'], self.batch_size),
                               train_inputs_neg_d: self.re_pad(X[u'd_aux'], self.batch_size),
                               train_input_q_weights: self.re_pad(X[u'idf'], self.batch_size),
                               input_mu: self.mus,
                               input_sigma: self.sigmas,
                               input_train_mask_pos: M_pos,
                               input_train_mask_neg: M_neg,
                               input_labels: Y}

            # Run the graph and fetch some of the nodes.
            _, l = sess.run([optimizer, loss], feed_dict=train_feed_dict)
            print l
            print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            print '[Train] @ iter: %d,' % (epoch * self.num_batch + batch_step),
            print 'train_loss: %.5f' % l
            if (batch_step == self.num_batch): break

    def valid_in_train(self, val_pair_file_path_list, train_inputs_q, train_inputs_pos_d, train_inputs_neg_d,
                       train_input_q_weights, input_mu, input_sigma, input_train_mask_pos, input_train_mask_neg,
                       sess, o_pos, epoch, loss, output):
        for filenum in range(len(val_pair_file_path_list)):
            scoredict = {}
            metricdict = {}
            # total_loss = 0
            batch_cnt = 0
            for BATCH in self.val_data_generator[filenum].pointwise_generate():  # val_pair_file_path_list[filenum], self.batch_size, with_idf=True):
                batch_cnt = batch_cnt + 1
                X_val, Y_val = BATCH
                M_pos = self.gen_mask(X_val[u'q'], X_val[u'd'])
                M_neg = self.gen_mask(X_val[u'q'], np.zeros([self.batch_size, self.max_d_len]))
                val_feed_dict = {train_inputs_q: self.re_pad(X_val[u'q'], self.batch_size),
                                 train_inputs_pos_d: self.re_pad(X_val[u'd'], self.batch_size),
                                 train_inputs_neg_d: self.re_pad(np.zeros([self.batch_size, self.max_d_len]),
                                                                 self.batch_size),
                                 train_input_q_weights: self.re_pad(X_val[u'idf'], self.batch_size),
                                 input_mu: self.mus,
                                 input_sigma: self.sigmas,
                                 input_train_mask_pos: M_pos,
                                 input_train_mask_neg: M_neg}

                o_p = sess.run(o_pos, feed_dict=val_feed_dict)
                # total_loss += l
                for num, i in enumerate(X_val['qid']):
                    if not scoredict.has_key(X_val['qid'][num]):
                        scoredict[X_val['qid'][num]] = {}
                    scoredict[X_val['qid'][num]][X_val['uid'][num]] = (o_p[num], Y_val[num])
                    if (output!=None):
                        output.write('%s\t%s\t%s\t%s\n' % (i, X_val['uid'][num], Y_val[num], o_p[num]))
            for q in scoredict.keys():
                y_pred = []
                y_label = []
                for doc in scoredict[q].keys():
                    pred, label = scoredict[q][doc]
                    y_pred.append(pred)
                    y_label.append(label)
                metrics = calc_metric(self.metriclist, y_pred, y_label)
                for i in metrics.keys():
                    if i not in metricdict.keys():
                        metricdict[i] = 0
                    metricdict[i] += metrics[i]
            # outstr = "Validation_"+str(filenum)+" METRICS: "
            assert batch_cnt > 0
            # valid_loss = total_loss / batch_cnt
            # outstr += "loss: %.5f "%(total_loss/batch_cnt ) ############
            # for i in metricdict.keys():
            #     metricdict[i]/=len(scoredict.keys())
            #     outstr += str(i) + ": " + str(metricdict[i])+ ';\t'
            # print outstr

            print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            print '[Eval] @ epoch: %d,' % (epoch + 1),
            # print 'valid loss: %.5f' % valid_loss,
            print ', '.join(['%s: %.5f' % (k, metricdict[k] / len(scoredict)) for k in metricdict])

    def test_in_train(self, train_inputs_q, train_inputs_pos_d, train_inputs_neg_d, train_input_q_weights,
                      input_mu, input_sigma, input_train_mask_pos, input_train_mask_neg,
                      sess, o_pos, epoch, output):
        scoredict = {}
        metricdict = {}
        for BATCH in self.test_data_generator.pointwise_generate():  # test_pair_file_path, self.batch_size, with_idf=True):
            X_test, Y_test = BATCH
            M_pos = self.gen_mask(X_test[u'q'], X_test[u'd'])
            M_neg = self.gen_mask(X_test[u'q'], np.zeros([self.batch_size, self.max_d_len]))
            test_feed_dict = {train_inputs_q: self.re_pad(X_test[u'q'], self.batch_size),
                              train_inputs_pos_d: self.re_pad(X_test[u'd'], self.batch_size),
                              train_inputs_neg_d: self.re_pad(np.zeros([self.batch_size, self.max_d_len]),
                                                              self.batch_size),
                              train_input_q_weights: self.re_pad(X_test[u'idf'], self.batch_size),
                              input_mu: self.mus,
                              input_sigma: self.sigmas,
                              input_train_mask_pos: M_pos,
                              input_train_mask_neg: M_neg}

            o_p = sess.run(o_pos, feed_dict=test_feed_dict)
            for num, i in enumerate(X_test['qid']):
                if not scoredict.has_key(X_test['qid'][num]):
                    scoredict[X_test['qid'][num]] = {}
                scoredict[X_test['qid'][num]][X_test['uid'][num]] = (o_p[num], Y_test[num])
                if (output!= None):
                    output.write('%s\t%s\t%s\t%s\n' % (i, X_test['uid'][num], Y_test[num], o_p[num]))###WRONG!!!###
            
        for q in scoredict.keys():
            y_pred = []
            y_label = []
            for doc in scoredict[q].keys():
                pred, label = scoredict[q][doc]
                y_pred.append(pred)
                y_label.append(label)
            metrics = calc_metric(self.metriclist, y_pred, y_label)
            for i in metrics.keys():
                if i not in metricdict.keys():
                    metricdict[i] = 0
                metricdict[i] += metrics[i]
        print '[%s]' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        print '[Eval] @ epoch: %d,' % (epoch + 1),
        # print 'valid loss: %.5f' % valid_loss,
        print ', '.join(['%s: %.5f' % (k, metricdict[k] / len(scoredict)) for k in metricdict])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--device", default='')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train_size", '-z', type=int, help="number of train samples")
    parser.add_argument("--load_model", '-l', action='store_true')
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_size", type=int, default=0)
    parser.add_argument("--output_score_file", '-o')
    parser.add_argument("--emb_file_path", '-e')
    #parser.add_argument("--checkpoint_dir", '-s', help="store data to here")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    conf = PyFileConfigLoader(args.config).load_config()
    print '\n##### CONFIG #####'
    with open(args.config) as config_file:
        for line in config_file:
            line = line.strip()
            if len(line) < 1:
                continue
            print line
    print '##### CONFIG #####\n'
    if args.train:
        nn = Knrm(config=conf)
        nn.train(train_size=args.train_size,
                 #checkpoint_dir=args.checkpoint_dir,
                 load_model=args.load_model)
        # else:
        #     nn = Knrm(config=conf)
        #     nn.test(test_point_file_path=args.test_file,
        #             test_size=args.test_size,
        #             output_file_path=args.output_score_file,
        #             load_model=True)
        #     checkpoint_dir=args.checkpoint_dir)
