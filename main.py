from collections import defaultdict
import matplotlib.image as mpimg
import matplotlib.pyplot as pl
import matplotlib as plt
import numpy as np
import random
import copy
import sys
import re
import os

from functions import Cost, Activation
from neural_network import neural_network
# from rbm import RBM, test_rbm

from MLP import MLP
from Functions1 import *
from Distances import *

def input_data():
    
    data_folder = './data/'
    infilenames = [ fn for fn in os.listdir(data_folder) if os.path.isfile( os.path.join(data_folder, fn) ) ]

    n_classes = 26
    X_test  = []  # TEST
    y_test  = []
    X_train = [] # * n_classes # TRAIN
    y_train = [] # * n_classes

    y = [0] * 26
    k, z = 0, 0
    for fn in infilenames:
        match_X0 = re.match(ur'class-(\d+).bmp', fn)
        match_X1 = re.match(ur'mutant-(\d+)-[012345]-0.bmp', fn)
        match_T  = re.match(ur'mutant-(\d+)-[678]-0\.bmp', fn)

        image = mpimg.imread(os.path.join('./data/', fn))
        b = np.array(image[:,:,0] > 0, dtype=int).flatten()

        if   match_X0 is not None:
            i = int(match_X0.group(1))
            r = copy.deepcopy(y); r[i] = 1
            k += 1

            y_train.append(r)
            X_train.append(b)
            # i = int(match_X.group(1))
            # y_train[i] = 1 if i == 1 else 0
            # X_train[i] = np.array(image).flatten()
        elif match_X1 is not None:
            i = int(match_X1.group(1))
            r = copy.deepcopy(y); r[i] = 1
            k += 1

            y_train.append(r)
            X_train.append(b)
        elif match_T  is not None:
            i = int(match_T.group(1))
            r = copy.deepcopy(y); r[i] = 1
            z += 1

            y_test.append(r)
            X_test.append(b)

    # inputs by rows
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print X_test.shape, y_test.shape, X_train.shape, y_train.shape
    
    return X_train, y_train, X_test, y_test

class Cost1:
    def __init__(self):
        self.F =   log_Bernoulli_likelihood
        self.D = d_log_Bernoulli_likelihood


def test_ann():

    X_train, y_train, X_test, y_test = input_data()

    # X_train = np.array([ [0,1,1,0,0],
    #                      [0,1,1,1,0],
    #                      [0,1,1,1,1],
    #                      [1,1,1,0,0],
    #                      [1,1,0,0,0] ])
    # y_train = np.array([0, 0, 0, 1, 1]).reshape((5,1))


    layers = [X_train.shape[1],  30, y_train.shape[1]]
    print layers, '\n'

    mlp = MLP(X_train.shape[1], [100, y_train.shape[1]],
              [sigmoid, sigmoid]
              # [Activation.sigmoid_function,
              #  Activation.sigmoid_function]
              )
              
    cost, test, accr = 10., 0., 0.
    while cost > 0.1:
        cost = mlp.train_backprop(X_train, y_train,
                                  goal=log_Bernoulli_likelihood,
                                  # goal = Cost.Bernoulli_function,
                                 d_goal = d_log_Bernoulli_likelihood,
                                 # d_goal = Cost.Bernoulli_derivative,
                                d_f_list = [d_sigmoid, d_sigmoid],
                                # d_f_list = [Activation.sigmoid_derivative,
                                #             Activation.sigmoid_derivative],
                                cv_input_data = X_test,
                                cv_output_data= y_test,
                                max_iter = 10000,

                                verbose=True)
    
    # ann = neural_network(layers, [0] + [ Activation() ] * 2 , Cost1())
    # 
    # cost, test, accr = 10., 0., 0.
    # while cost > 0.1:
    # 
    #     # test = ann.evaluate(X_test,  y_test)
    #     cost = ann.evaluate(X_train, y_train)
    #     accr = ann.accuracy(X_train, y_train)
    #     print 'Train cost=%.3f Test cost=%.3f Accuracy=%3f' % (cost, test, accr)
    #     ann.backprop(X_train, y_train)
    # 
    ## print 'Train cost=%.3f Test cost=%.3f Accuracy=%3f' % (cost, test, accr)
    #return


def test_rbm():
    # rng = np.random.RandomState(123)
    # rbm = RBM(input=X_train, n_visible=X_train.shape[0], n_hidden=X_train.shape[1], rng=rng)
    # 
    # for i in xrange(iters):
    #     rbm.contrastive_divergence(k=1)
    #   
    #     cst = rbm.get_reconstruction_cross_entropy()
    #     print 'Training iter %d, cost is %s' % (i, cst)
    #    
    # # v = numpy.array([[1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0]])
    # print rbm.reconstruct(X_test)
    return


if __name__ == '__main__':
    test_ann()
    test_rbm()

