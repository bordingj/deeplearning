# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 13:49:03 2015

@author: bordingj
"""

import models
import numpy as np
import theano.tensor as T
import theano
from layers import relu
from training import train_model, train

def load_mnist():
    import cPickle
    import gzip
    dataset = "mnist.pkl.gz"
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    n_labels = len(np.unique(train_set[1]))
    output = (train_set, valid_set, test_set)
    f.close()

    return output, n_labels

if __name__=="__main__":

    datasets, num_labels = load_mnist()
    train_set_x, train_set_t = datasets[0]
    train_set_x = np.asarray(train_set_x, dtype=np.float32)
    train_set_y = np.asarray(train_set_t, dtype=np.int32) # add one to each label such that 0 is not a label

    valid_set_x, valid_set_t = datasets[1]
    valid_set_x = np.asarray(valid_set_x, dtype=np.float32)
    valid_set_y = np.asarray(valid_set_t,dtype=np.int32)

    test_set_x, test_set_t = datasets[2]
    test_set_x = np.asarray(test_set_x, dtype=np.float32)
    test_set_y = np.asarray(test_set_t, dtype=np.int32)

    N_features = valid_set_x.shape[1]
    N_hidden = 500
    N_labels = len(np.unique(valid_set_y))
    seed = 123

    theano.config.mode = 'FAST_RUN'
    #import theano.sandbox.cuda
    #theano.sandbox.cuda.use("gpu0")

    classifier = models.MLP(
                        X=theano.shared(valid_set_x,borrow=True),
                        y=theano.shared(valid_set_y,borrow=True),
                        N_features=N_features,
                        N_hidden=N_hidden,
                        N_labels=N_labels,
                        activationFunction=T.tanh,
                        seed=123)

    tr_model = train_model(classifier=classifier,
                           X=train_set_x,
                           y=train_set_y,
                           X_valid=valid_set_x,
                           y_valid=valid_set_y,
                           L1_reg=0.01,L2_reg=0.0)

    N = train_set_x.shape[0]
    train(tr_model,
          max_epoch=1000,
          batch_size=4000,
          N=N)