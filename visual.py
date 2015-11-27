import matplotlib.pyplot as pl
import matplotlib as plt
import numpy as np

from test_ann import test_ann
from test_rbm import test_rbm


def ann_visual(X_train, y_train, X_test, y_test):

    xes, ycv, y_c = range(1,6), [], []
    for m in xes:
        
        layers = [ X_train.shape[1] ] + [ 30 ] * m + [ y_train.shape[1] ]
        cost, cv_cost = test_ann(X_train, y_train, X_test, y_test, layers, 1000)
        
        y_c.append(cost)
        ycv.append(cv_cost)

    pl.plot(xes, y_c, 'r-')
    pl.plot(xes, ycv, 'b-')
    pl.show()

    return


def rbm_visual(X_train, y_train, X_test, y_test):

    rbm = test_rbm(X_train, y_train, X_test, y_test)
       
    pl.figure(figsize=(50,50))
    W = rbm.W.T
    print W.shape

    for i,w in enumerate(W):
        w = (np.array(w) - np.mean(w)) / np.var(w) / len(w)
        pl.subplot(10, 10, i+1)
        pl.imshow( w.reshape( (29,29) ) )
        if not (i%50): print '.'

    pl.show()
