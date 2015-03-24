# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 12:46:06 2015

@author: bordingj
"""
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

def relu(x):
    return (x > 0)*x

class HiddenLayer(object):
    def __init__(self, Z, W, b, activationFunction=relu):
        #self.input = input

        self.W = W
        self.b = b

        self.output = (
            T.dot(self.input, self.W) + self.b if activationFunction is None
            else activationFunction(T.dot(Z, self.W) + self.b)
        )


class SoftMaxLayer(object):
    def __init__(self, Z, W, b):
        #self.input = input

        self.W = W
        self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(Z, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

class LeNetConvPoolLayer(object):
    def __init__(self, Z, W, b, poolsize, activationFunction):

        self.W = W
        self.b = b
        conv_output = conv.conv2d(input=Z, filters=self.W)

        pooled_out = T.signal.downsample.max_pool_2d(
            input=conv_output,
            ds=poolsize,
            ignore_border=True)

        self.output = activationFunction(
                            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
                            )