from math import *
import numpy as np
import random
import sys


def identity(x): return x


class Cost(object):
    """ """
    def __init__(self, type='Bernoulli'):
        self.type = type
        if   type == 'Bernoulli':
            self.F = Cost.Bernoulli_function
            self.D = Cost.Bernoulli_derivative
        elif type == 'square':
            self.F = Cost.square_function
            self.D = Cost.square_derivative
        elif type == 'hamming':
            self.F = Cost.hamming_function
            self.D = Cost.hamming_derivative
        else:
            raise ValueError

    @staticmethod
    def support_functions():
        return [ 'Bernoulli', 'square', 'hamming' ]

    @staticmethod
    @np.vectorize
    def likelihood(y, h):
      res, eps = 0., np.finfo(float).eps
      # ---------------------------
      y0, y1 = np.fabs(y - 0.) < eps, np.fabs(y - 1.) < eps
      h0, h1 = np.fabs(h - 0.) < eps, np.fabs(h - 1.) < eps
      # ---------------------------
      res  = eps if (y0 or h1) else (    y  * np.log(    h  if (not h0) else eps))
      res += eps if (y1 or h0) else ((1.-y) * np.log((1.-h) if (not h1) else eps))
      # ---------------------------
      return res
    
    @staticmethod
    def Bernoulli_function(y, h):
        """ -sum( y * np.log(h) - (1. - y) * np.log(1. - h) ) """
        return  - np.sum( Cost.likelihood(y, h), axis=1 )

    @staticmethod
    @np.vectorize
    def Bernoulli_derivative(y, h):
        """ (y / h) - (1. - y) / (1. - h) """
        # ---------------------------
        eps = np.finfo(float).eps
        h0, h1 = np.fabs(h-0.) < eps, np.fabs(h-1.) < eps
        # ---------------------------
        res  = ((1.-y)/(1.-h) if (not h1) else (1.-y)/eps)
        res -= (    y / h     if (not h0) else     y /eps)
        # ---------------------------
        return res

    @staticmethod
    def square_function(y, h):
        return np.sum(np.power(y - h, 2), axis=1) / 2.

    @staticmethod
    def square_derivative(y, h):
        return y - h;

    @staticmethod
    def hamming_function(y, h):
        return np.sum(np.abs(y - h), axis=1)

    @staticmethod
    def hamming_derivative(y, h):
        return np.sign(y - h)


class Activation(object):
    """ """
    def __init__(self, type='sigmoid'):
        """ """
        self.type = type
        if   type == 'identity':
            self.F = identity
            self.D = np.ones_like
        elif type == 'sigmoid':
            self.F = Activation.sigmoid_function
            self.D = Activation.sigmoid_derivative
        elif type == 'hyperbolic':
            self.F = Activation.hyperbolic_function
            self.D = Activation.hyperbolic_derivative
        elif type == 'softmax':
            self.F = Activation.softmax_function
            self.D = Activation.softmax_derivative
        elif type == 'ReLU':
            self.F = Activation.ReLU_function
            self.D = Activation.ReLU_derivative
        elif type == 'rectifier':
            self.F = Activation.rectifier_function
            self.D = Activation.rectifier_derivative
        else:
            raise ValueError

    @staticmethod
    def support_functions():
        return [ 'identity', 'sigmoid', 'hyperbolic',
                 'softmax',  'ReLU',    'rectifier' ]

    @staticmethod
    def sigmoid_function(x, a=1.):
        return 1. / ( 1. + np.exp(-a * x) )

    @staticmethod
    def sigmoid_derivative(x, a=1.):
        s = Activation.sigmoid_function(x, a)
        return  a * s * (np.ones_like(s) - s)

    @staticmethod
    def hyperbolic_function(x, a=1.):
        """ np.exp(a * x) - np.exp(-a * x) 
            ------------------------------
            np.exp(a * x) + np.exp(-a * x)
        """
        return np.tanh(a * x)

    @staticmethod
    def hyperbolic_derivative(x, a=1.):
        h = Activation.hyperbolic_function(x, a)
        return  a * (1. - h * h)

    @staticmethod
    def softmax_function(x):
        """ exp(m_i) / sum_i( exp(m_i) ), i is index of dimension """
        e = np.exp(x) # (x - np.max(x)) # prevent overflow
        return e / np.sum(e)

    @staticmethod
    def softmax_derivative(x):
        s = Activation.softmax_function(x)
        return s * (1. - s)

    @staticmethod
    def ReLU_function(x):
        return  x  * (x > 0.)

    @staticmethod
    def ReLU_derivative(x):
        return  1. * (x > 0.)

    @staticmethod
    def rectifier_function(x):
        return np.log( 1. + np.exp(x) )

    @staticmethod
    def rectifier_derivative(x):
        return  Activation.sigmoid_function(x)


class Regularization(object):
    """ """
    def __init__(self, type='L2'):
        """ """
        self.type = type
        if   type == 'L0':
            self.F = identity
            self.D = np.ones_like
        elif type == 'L1':
            self.F = Regularization.L1_F
            self.D = Regularization.L1_D
        elif type == 'L2':
            self.F = Regularization.L2_F
            self.D = identity
        else:
            raise Exception('Supported functions: L0, L1, L2')

    @staticmethod
    def support_functions(): return [ 'L0', 'L1', 'L2' ]
    
    @staticmethod
    def L1_F(x):  return (x > 0) * 2 - 1.0  # sign(x) = x / abs(x)

    @staticmethod
    def L1_D(x):  return  np.sum(np.abs( x.reshape( (1, np.prod(x.shape)) ) ))

    @staticmethod
    def L2_F(x):  return  np.sum(np.power( x.reshape( (1, np.prod(x.shape)) ), 2)) * 0.5

