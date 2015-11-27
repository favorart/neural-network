from math import *

import numpy as np
import random
import time
import sys
import re

from ann import NeuralNetwork
from funcs import Cost, Activation, Regularization


def test_ann(X_train, y_train, X_test, y_test, layers, max_iter=100):
    """ """
    nn = NeuralNetwork(layers, [ Activation() ] * (len(layers) - 1), 
                       # regularization_rate=0.0001,
                       # regularization=Regularization('L2'),
                       cost=Cost('Bernoulli') )

    # print layers, ', ', nn.neurons[1].type, \
    #               ', ', nn.regular.type, \
    #               ', ', nn.cost.type, '\n' # 

    cost, cv_cost = 10., 0.
    accr, cv_accr =  0., 0.
    cv_cost_prev  =  0.

    # tolerance = np.finfo(float).eps
    cost_threshold = 0.1 # min_cost: np.finfo(float).eps
    stop_threshold = 0.05

    total_time = time.time()
    start_time = time.time()

    iter, n_iter_skip = 0, 100
    # iter, max_iter, n_iter_skip = 0, 1000, 100
    # while cost > cost_threshold:
    while iter < max_iter:

        Pt = nn.predict_matrix(X_test)
        # cv_cost = nn.calculate_cost(X_test,  y_test, Pt)
        cv_accr = nn.accuracy(X_test, y_test, Pt)
        
        P = nn.predict_matrix(X_train)
        # cost = nn.calculate_cost(X_train, y_train, P)
        accr = nn.accuracy(X_train, y_train, P)
        
        if  not (iter % n_iter_skip):
            print '%4d\ttime=%.3f sec.\tCost(X=%.3f T=%.3f)\tAccr(X=%.3f T=%.3f)' % \
                  (iter, time.time() - start_time, cost, cv_cost, accr, cv_accr)
            start_time = time.time()

            # # Ранняя остановка на основании ошибки на валидационном множестве
            # if cv_cost_prev and (cv_cost - cv_cost_prev) > stop_threshold:
            #     break
            # else: cv_cost_prev = cv_cost

        # if iter > max_iter: break
        nn.backprop(X_train, y_train, P)
        iter += 1
    
    print '\n%4d\ttime=%.3f sec.\tCost(X=%.3f T=%.3f)\tAccr(X=%.3f T=%.3f)' % \
            (iter, time.time() - total_time, cost, cv_cost, accr, cv_accr)
    return accr, cv_accr # cost, cv_cost

