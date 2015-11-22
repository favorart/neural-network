import numpy as np
import random
import sys
import re

from functions import Cost, Activation, Regularization
from rbm import BoltzmannMachine


def test_rbm(X_train, y_train, X_test, y_test):
    """ """
    rnd = np.random.RandomState(123)
    rbm = BoltzmannMachine(X_train, n_visible=X_train.shape[0], n_hidden=X_train.shape[1], rnd=rnd)
    
    iters = 1000
    for i in xrange(iters):
        rbm.contrastive_divergence(k=1)
      
        cst = rbm.get_reconstruction_cross_entropy()
        print 'Training iter %d, cost is %s' % (i, cst)
       
    print rbm.reconstruct(X_test)
    return

