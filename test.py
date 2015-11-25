from math import *

import numpy as np
import random
import time
import sys
import re

from nn import NeuralNetwork
from functions import Cost, Cost1, Activation, Regularization


def test_ann(X_train, y_train, X_test, y_test):
    """ """
    layers = [X_train.shape[1],  30, y_train.shape[1]]
    nn = NeuralNetwork(layers, [ Activation(), Activation() ], cost=Cost1(),)
    print layers, '\n'

    threshold = 0.001
    threshold_stop = 0.1

    cost, test, test_prev, accr, it = 10., 0., 0., 0., 0
    while cost > threshold:
    
        test = nn.calculate_cost(X_test,  y_test)
        # if not test_prev: test_prev = test

        P = nn.predict_matrix(X_train)
        cost = nn.calculate_cost(X_train, y_train, P)
        accr = nn.accuracy(X_train, y_train, P)
        
        if not (it%100):
            print 'Cost (train=%.3f test=%.3f)\tAccuracy=%3f' % (cost, test, accr)

        # Ранняя остановка на основании ошибки на валидационном множестве
        # if fabs(test - test_prev) > threshold_stop:
        #     break
        # else: test_prev = test

        nn.backprop(X_train, y_train, P)
        it += 1
    
    print 'Cost (train=%.3f test=%.3f)\tAccuracy=%3f' % (cost, test, accr)
    return

