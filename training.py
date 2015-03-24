# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 19:44:23 2015

@author: bordingj
"""
from scipy.optimize import minimize
import numpy as np
import datetime
import math
import theano
import theano.tensor as T
from theano.misc import gnumpy_utils
from theano import sandbox
import gnumpy as gpu
from optimize.gnumpy_optimize import fmin_cg, fmin_steepest_descent

class train_model(object):
    def __init__(self, classifier, X, y, X_valid, y_valid, L1_reg, L2_reg):
        self.y = y
        self.X = X
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.classifier = classifier
        self.cost = ( self.classifier.negative_log_likelihood().mean()
                    + L1_reg * self.classifier.L1
                    + L2_reg * self.classifier.L2_sqr )
                    
        self.cost_Tfunc = theano.function([],self.cost)
        
        self.grad = T.grad(self.cost, self.classifier.flat_params)
        self.validate_Tfunc = theano.function([],self.classifier.errors())       
        params = T.fvector()
        self.grad_Tfunc = theano.function([params],
                                           sandbox.cuda.basic_ops.gpu_from_host(self.grad),
                                          givens = [(self.classifier.flat_params,
                                                        params)])
        
        self.grad_Tfunc2 = theano.function([],self.grad)
        
        self.get_cost_at_x = theano.function([params],self.cost,
                                             givens = [(self.classifier.flat_params,
                                                        params)])
                
    
    def ObjFun(self, params):
        params = gnumpy_utils.garray_to_cudandarray(params)
        return self.get_cost_at_x(params)
    
    def ObjFunPrime(self, params):
        params = gnumpy_utils.garray_to_cudandarray(params)
        grad = self.grad_Tfunc(params)
        return gnumpy_utils.cudandarray_to_garray(grad)
        
    def validate_model(self, on_train=False):
        self.classifier.X.set_value(self.X_valid,borrow=True)
        self.classifier.y.set_value(self.y_valid,borrow=True)
        return self.validate_Tfunc()

    def set_random_batch(self,indices):
        self.classifier.X.set_value(self.X[indices,:],borrow=True)
        self.classifier.y.set_value(self.y[indices],borrow=True)

    def cost_and_grad_fun(self, param_values, grad_weight=1):
        self.classifier.flat_params.set_value(param_values, borrow=True)
        return self.cost_Tfunc(), grad_weight*self.grad_Tfunc2()
    
    
    def Run_fmin_GD(self, maxiter):
        x0 = self.classifier.flat_params.get_value(borrow=True, 
                                                   return_internal_type=True)
        x0 = gnumpy_utils.cudandarray_to_garray(x0)
        f_op, params_opt = fmin_steepest_descent(f=self.ObjFun, x0=x0, 
                                   fprime=self.ObjFunPrime,
                                   maxiter=maxiter)
        params_opt = gnumpy_utils.garray_to_cudandarray(params_opt)
        
        self.classifier.flat_params.set_value(params_opt)

    def Run_fmin_cg(self, maxiter):
        x0 = self.classifier.flat_params.get_value(borrow=True, 
                                                   return_internal_type=True)
        x0 = gnumpy_utils.cudandarray_to_garray(x0)
        f_op, params_opt = fmin_cg(f=self.ObjFun, x0=x0, 
                                   fprime=self.ObjFunPrime,
                                   maxiter=maxiter)
        params_opt = gnumpy_utils.garray_to_cudandarray(params_opt)
        
        self.classifier.flat_params.set_value(params_opt)
        
def train(train_model, max_epoch, batch_size, N, max_number_of_ascends=None,
          iter_at_each_epoch=None):
    assert batch_size >= 500, 'batch size must be at least 1000'
    if iter_at_each_epoch == None:
        iter_at_each_epoch = math.floor((math.ceil(batch_size/1000.0)+np.log(batch_size))/2.0)
        print 'performing %d CG iterations every epoch' % (iter_at_each_epoch)

    n_train_batches = N / batch_size
    
    start = datetime.datetime.now()
    error_current = train_model.validate_model()
    error_old = error_current
    best_error = error_current
    best_theta = train_model.classifier.flat_params.get_value(borrow=True)
    k = 0
    number_of_ascends = 0
    if max_number_of_ascends==None:
        max_number_of_ascends = n_train_batches*10;
        print ("Optimizing using scipy.optimize.minimize with stochastic minibatch CG... ")

    best_k = 0

    while max_epoch >= k and max_number_of_ascends >= number_of_ascends :
        k += 1
        indices = np.random.randint(low=0, high=N, size=batch_size)
        train_model.set_random_batch(indices)
        OptimizeResult = minimize(fun=train_model.cost_and_grad_fun,\
                                      x0=train_model.classifier.flat_params.get_value(borrow=True),\
                                      method="CG",\
                                      jac=True,\
                                      callback=None,\
                                      options={
                                          "maxiter": iter_at_each_epoch,
                                          "disp": False
                                          }
                                          )
        error_old = error_current
        train_model.classifier.flat_params.set_value(OptimizeResult.x.astype(np.float32), borrow=True)
        error_current = train_model.validate_model()
        print('Validation error %f %% at epoch number %d' % (error_current*100, k))
        if error_current >= best_error:
            number_of_ascends += 1
        else:
            number_of_ascends = 0
            best_theta = train_model.classifier.flat_params.get_value(borrow=True)
            best_k = k
            best_error = error_current
          
    train_model.classifier.flat_params.set_value(best_theta, borrow=True)
    finish = datetime.datetime.now()
    print 'Training took %.1f seconds' % ((finish - start).seconds)
    print('Best Validation error %f %% at iteration number %d' % (best_error*100, best_k))
    print 'performed %d CG iterations every epoch' % (iter_at_each_epoch)
    return best_theta, best_k

def train2(train_model, max_epoch, batch_size, N, method="CG", max_number_of_ascends=None,
          iter_at_each_epoch=None):
              
    assert batch_size >= 500, 'batch size must be at least 1000'
    
    if iter_at_each_epoch == None:
        iter_at_each_epoch = math.floor((math.ceil(batch_size/1000.0)+np.log(batch_size))/2.0)
        print 'performing %d CG iterations every epoch' % (iter_at_each_epoch)

    n_train_batches = N / batch_size
    
    
    start = datetime.datetime.now()    
    error_current = train_model.validate_model()
    error_old = error_current
    best_error = error_current
    best_theta = train_model.classifier.flat_params.get_value(borrow=True,
                                                              return_internal_type=True)
    k = 0
    number_of_ascends = 0
    if max_number_of_ascends==None:
        max_number_of_ascends = n_train_batches*10;
        print ("Optimizing using scipy.optimize.minimize with stochastic minibatch CG... ")
    best_k = 0
    time_spend_copying = 0
    time_spend_computing = 0

    while max_epoch >= k and max_number_of_ascends >= number_of_ascends :
        k += 1
        indices = np.random.randint(low=0, high=N, size=batch_size)
        train_model.set_random_batch(indices)
        if method=="CG":
            train_model.Run_fmin_cg(maxiter=iter_at_each_epoch)
        else:
            train_model.Run_fmin_GD(maxiter=iter_at_each_epoch)
        
        error_old = error_current
        error_current = train_model.validate_model()
        print('Validation error %f %% at epoch number %d' % (error_current*100, k))
        if error_current >= best_error:
            number_of_ascends += 1
        else:
            number_of_ascends = 0
            best_theta = train_model.classifier.flat_params.get_value(borrow=True,
                                                                      return_internal_type=True)
            best_k = k
            best_error = error_current
        
                
    train_model.classifier.flat_params.set_value(best_theta, borrow=True)
    finish = datetime.datetime.now()
    print 'Training took %.1f seconds' % ((finish - start).seconds)
    print('Best Validation error %f %% at iteration number %d' % (best_error*100, best_k))
    print 'performed %d CG iterations every epoch' % (iter_at_each_epoch)
    
    return best_theta, best_k