# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 13:17:19 2015

@author: bordingj
"""

from layers import *
import theano.tensor as T

def generateRandomWeights(dim, seed=None):
    rng = np.random.RandomState(seed)
    n_row = dim[0]
    n_col = dim[1]
    r = np.sqrt(6) / np.sqrt(n_row + n_col)
    return np.asarray(rng.uniform( low=-r,high=r, \
                        size=(n_row,n_col) ),dtype=np.float32 )

def generateRandomConvWeights(dim, filter_shape, seed=None):
    rng = np.random.RandomState(seed)
    n_row = dim[0]
    n_col = dim[1]
    r = np.sqrt(6) / np.sqrt(n_row + n_col)
    return np.asarray(rng.uniform( low=-r,high=r, \
                        size=filter_shape ),dtype=np.float32 )


class MLP(object):
    def __init__(self, X, y, N_features, N_hidden, N_labels, activationFunction=relu, seed=None,
                W_hidden_values=None, b_hidden_values=None,
                W_softmax_values=None, b_softmax_values=None):

        self.X = X
        self.y = y

        #Initialize random weights for Hidden Layer
        if W_hidden_values is None:
            W_hidden_values = generateRandomWeights((N_features, N_hidden), seed)
            if activationFunction == T.nnet.sigmoid or \
                activationFunction == T.nnet.hard_sigmoid or \
                activationFunction == T.nnet.ultra_fast_sigmoid:
                W_hidden_values *= 4
        if b_hidden_values is None:
            b_hidden_values = np.zeros((N_hidden,), dtype=np.float32)


        #Initialize weights for softmax layer
        if W_softmax_values is None:
            W_softmax_values = np.zeros((N_hidden, N_labels), dtype=np.float32)

        if b_softmax_values is None:
            b_softmax_values = np.zeros((N_labels,), dtype=np.float32)


        self.flat_params = theano.shared(np.concatenate(map(lambda x: x.flatten(),
                                             [W_hidden_values, b_hidden_values,
                                              W_softmax_values, b_softmax_values])
                                              ),
                                        borrow=True)

        offset = 0
        W_hidden_n = N_features*N_hidden
        self.hiddenLayer = HiddenLayer(Z=self.X,
                W=self.flat_params[offset:(offset+ W_hidden_n)].reshape((N_features,N_hidden)),
                b=self.flat_params[(offset + W_hidden_n):(offset + W_hidden_n + N_hidden)],
                activationFunction=activationFunction)

        offset += W_hidden_n+N_hidden
        W_softmax_n = N_hidden*N_labels
        self.softMaxLayer = SoftMaxLayer(Z=self.hiddenLayer.output,
                W=self.flat_params[offset:(offset + W_softmax_n)].reshape((N_hidden,N_labels)),
                b=self.flat_params[(offset + W_softmax_n):])

        self.L1 = 1.0/self.X.shape[0]*(
            abs(self.hiddenLayer.W).sum()
            + abs(self.softMaxLayer.W).sum()    )

        self.L2_sqr = 1.0/self.X.shape[0]*(
            (self.hiddenLayer.W ** 2).sum()
            + (self.softMaxLayer.W ** 2).sum()    )

    def negative_log_likelihood(self):
        return -T.mean(T.log(self.softMaxLayer.p_y_given_x)[T.arange(self.y.shape[0]), self.y])

    def errors(self):
        # check if y has same dimension of y_pred
        if self.y.ndim != self.softMaxLayer.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', self.y.type, 'y_pred', self.softMaxLayer.y_pred.type)
            )
        # check if y is of the correct datatype
        if self.y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.softMaxLayer.y_pred, self.y))
        else:
            raise NotImplementedError()

class CNN(object):
    def __init__(self, X, y, image_size, filter_shape, nkerns,
                 N_hidden, N_labels, poolsize=(2,2),
                activationFunctions=(relu,relu,relu),
                seed=None):

        self.X = X
        self.y = y

        #Initialize random weights for first Convolutional Layer
        filter1_shape = (nkerns[0],1) + filter_shape
        fan1_in = np.prod(filter1_shape[1:])
        fan1_out = (filter1_shape[0] * np.prod(filter1_shape[2:]) / np.prod(poolsize))
        W_filter1_values = generateRandomConvWeights(dim=(fan1_in,fan1_out),
                                                     filter_shape=filter1_shape,
                                                     seed=seed)
        if activationFunctions[0] == T.nnet.sigmoid or \
            activationFunctions[0] == T.nnet.hard_sigmoid or \
            activationFunctions[0] == T.nnet.ultra_fast_sigmoid:
            W_filter1_values *= 4
        b_filter1_values = np.zeros((filter1_shape[0],), dtype=np.float32)

        #Initialize random weights for Second Convolutional Layer
        filter2_shape = (nkerns[1], nkerns[0]) + filter_shape
        fan2_in = np.prod(filter2_shape[1:])
        fan2_out = (filter2_shape[0] * np.prod(filter2_shape[2:]) / np.prod(poolsize))
        W_filter2_values = generateRandomConvWeights(dim=(fan2_in,fan2_out),
                                                     filter_shape=filter2_shape,
                                                     seed=seed)
        if activationFunctions[1] == T.nnet.sigmoid or \
            activationFunctions[1] == T.nnet.hard_sigmoid or \
            activationFunctions[1] == T.nnet.ultra_fast_sigmoid:
            W_filter2_values *= 4
        b_filter2_values = np.zeros((filter2_shape[0],), dtype=np.float32)
        
        N_features = nkerns[1]*(filter_shape[0]-1)**2
        #Initialize random weights for Fully Connected Hidden Layer
        W_hidden_values = generateRandomWeights((N_features, N_hidden), seed)
        if activationFunctions[2] == T.nnet.sigmoid or \
            activationFunctions[2] == T.nnet.hard_sigmoid or \
            activationFunctions[2] == T.nnet.ultra_fast_sigmoid:
            W_hidden_values *= 4
        b_hidden_values = np.zeros((N_hidden,), dtype=np.float32)

        #Initialize weights for softmax layer
        W_softmax_values = np.zeros((N_hidden, N_labels), dtype=np.float32)
        b_softmax_values = np.zeros((N_labels,), dtype=np.float32)

        self.flat_params = theano.shared(np.concatenate(map(lambda x: x.flatten(),
                                             [W_filter1_values, b_filter1_values,
                                              W_filter2_values, b_filter2_values,
                                              W_hidden_values, b_hidden_values,
                                              W_softmax_values, b_softmax_values])
                                              ),
                                        borrow=True)

        self.X_4Dtensor = self.X.reshape( ( (self.X.shape[0],1) + image_size ) )
        offset = 0
        #build first Convolutional Layer
        W_filter1_len = np.prod(filter1_shape)
        self.Convlayer1 = LeNetConvPoolLayer(Z=self.X_4Dtensor,
                                        W=self.flat_params[offset:(offset+W_filter1_len)]\
                                        .reshape(filter1_shape),
                                        b=self.flat_params[(offset+W_filter1_len):(offset+\
                                        W_filter1_len+filter1_shape[0])],
                                        poolsize=poolsize,
                                        activationFunction=activationFunctions[0])
        offset += W_filter1_len+filter1_shape[0]
        #Build Second Convolutional Layer
        W_filter2_len = np.prod(filter2_shape)
        self.Convlayer2 = LeNetConvPoolLayer(Z=self.Convlayer1.output,
                                        W=self.flat_params[offset:(offset+W_filter2_len)]\
                                        .reshape(filter2_shape),
                                        b=self.flat_params[(offset+W_filter2_len):(offset+\
                                        W_filter2_len+filter2_shape[0])],
                                        poolsize=poolsize,
                                        activationFunction=activationFunctions[1])
        offset += W_filter2_len+filter2_shape[0]
        #Build Fully Connected Hidden Layer
        W_hidden_n = N_features*N_hidden
        self.hiddenLayer = HiddenLayer(Z=self.Convlayer2.output.reshape((self.X.shape[0],N_features)),
                W=self.flat_params[offset:(offset+ W_hidden_n)].reshape((N_features,N_hidden)),
                b=self.flat_params[(offset + W_hidden_n):(offset + W_hidden_n + N_hidden)],
                activationFunction=activationFunctions[2])

        offset += W_hidden_n+N_hidden
        W_softmax_n = N_hidden*N_labels
        self.softMaxLayer = SoftMaxLayer(Z=self.hiddenLayer.output,
                W=self.flat_params[offset:(offset + W_softmax_n)].reshape((N_hidden,N_labels)),
                b=self.flat_params[(offset + W_softmax_n):])


        self.L1 = (
            abs(self.Convlayer1.W).mean()
            +abs(self.Convlayer2.W).mean()
            +abs(self.hiddenLayer.W).mean()
            + abs(self.softMaxLayer.W).mean()    )

        self.L2_sqr = (
            (self.Convlayer1.W ** 2).mean()
            + (self.Convlayer2.W ** 2).mean()
            + (self.hiddenLayer.W ** 2).mean()
            + (self.softMaxLayer.W ** 2).mean()    )

    def negative_log_likelihood(self):
        return -T.mean(T.log(self.softMaxLayer.p_y_given_x)[T.arange(self.y.shape[0]), self.y])

    def errors(self):
        # check if y has same dimension of y_pred
        if self.y.ndim != self.softMaxLayer.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', self.y.type, 'y_pred', self.softMaxLayer.y_pred.type)
            )
        # check if y is of the correct datatype
        if self.y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.softMaxLayer.y_pred, self.y))
        else:
            raise NotImplementedError()
