import numpy as np
import random
import sys
import re

from functions import Cost, Activation, Regularization
from nn import NeuralNetwork


def test_ann(X_train, y_train, X_test, y_test):
    """ """
    layers = [X_train.shape[1],  30, y_train.shape[1]]
    nn = NeuralNetwork(layers, [ Activation(), Activation() ])
    print layers, '\n'

    threshold = 0.0001
    threshold_stop = 0.1

    cost, test, test_prev, accr, it = 10., 0., 0., 0., 0
    while cost > threshold:
    
        # Ранняя остановка на основании ошибки на валидационном множестве
        if fabs(test - test_prev) > threshold_stop:
            break
        else: test_prev = test

        test = nn.evaluate(X_test,  y_test)
        cost = nn.evaluate(X_train, y_train)
        accr = nn.accuracy(X_train, y_train)

        if not (it%100):
            print 'Train cost=%.3f\tTest cost=%.3f\tAccuracy=%3f' % (cost, test, accr)

        ann.backprop(X_train, y_train)
        it += 1
    
    print 'Train cost=%.3f\tTest cost=%.3f\tAccuracy=%3f' % (cost, test, accr)
    return

