import numpy as np
import random
import time
import sys
import re

from funcs import Cost, Activation, Regularization
from rbm import BoltzmannMachine

import matplotlib.pyplot as pl
import matplotlib as plt
import numpy as np


def test_rbm(X_train, y_train, X_test, y_test, max_iter=1000, verbose=True):
    """ """
    rnd = np.random.RandomState(123)
    rbm = BoltzmannMachine(n_visible=X_train.shape[1], 
                           n_hidden =100, # X_train.shape[0]
                           # regularization_rate=0.0001,
                           # regularization=Regularization('L2'),
                           cost=Cost('hamming'),
                           neural_local_gain=(0.005, 0.995, 0.001, 1000),
                           rnd=rnd,
                           margin=0.00001)

    print X_train.shape[1], X_train.shape[0], rbm.cost.type, '\n'

    stop_threshold = 0.1  # stop training   if (1.  - stop_threshold) * (current cost) > min(cost)
    cost_threshold = 0.1  # min_train_cost = np.finfo(float).eps, # minimum value of cost on train set
    # tolerance=0, # minimum step of cost

    total_time = time.time()
    start_time = time.time()

    iter, n_iter_skip = 0, 100
    
    cost = 10.
    while cost > cost_threshold:

        if  verbose:
            cost = rbm.evaluate_cost(X_train)

        if  verbose and not (iter % n_iter_skip):
            print '%4d\ttime=%.3f sec.\tCost_X=%.3f' % \
                  (iter, time.time() - start_time, cost)
            start_time = time.time()
        if  iter > max_iter: break

        rbm.contrastive_divergence(X_train, cd_k=1)
        iter += 1
    
    print '\n%4d\ttime=%.3f sec.\tCost_X=%.3f' % \
            (iter, time.time() - total_time, cost)

    h = np.array((np.random.uniform(0, 1, size=100) > 0), dtype=int).reshape((1,100))
    v = rbm.h_to_v(h, do_sampling=False)
    pl.imshow( (v > 0.5).reshape((29,29)) )
    pl.show()

    return rbm


if __name__ == "__main__":
    test_rbm()

