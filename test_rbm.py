import numpy as np
import random
import time
import sys
import re

from funcs import Cost, Activation, Regularization
from rbm import BoltzmannMachine


def test_rbm(X_train, y_train, X_test, y_test):
    """ """
    rnd = np.random.RandomState(123)
    rbm = BoltzmannMachine(n_visible=X_train.shape[1], 
                           n_hidden =100, # X_train.shape[0]
                           # regularization_rate=0.0001,
                           # regularization=Regularization('L2'),
                           cost=Cost('hamming'),
                           rnd=rnd,
                           margin=0.00001)

    print X_train.shape[1], X_train.shape[0], rbm.cost.type, '\n'

    stop_threshold = 0.1  # stop training if (1 - stop_threshold)*(current cost) > min(cost)
    cost_threshold = 0.1  # min_train_cost = np.finfo(float).eps, # minimum value of cost on train set
    # tolerance=0, # minimum step of cost

    total_time = time.time()
    start_time = time.time()

    cost, cv_cost = 10., 0.
    cv_prev_cost  =  0.
    
    iter, max_iter, n_iter_skip = 0, 1000, 100
    while cost > cost_threshold:

        # cv_cost = rbm.evaluate_cost(X_test)

        # cost = rbm.evaluate_cost(X_train)
        if  not (iter % n_iter_skip):
            print '%4d\ttime=%.3f sec.\tCost(X=%.3f T=%.3f)' % \
                  (iter, time.time() - start_time, cost, cv_cost)
        #     start_time = time.time()
        if iter > max_iter: break

        rbm.contrastive_divergence(X_train, cd_k=1)
        iter += 1
    
    print '\n%4d\ttime=%.3f sec.\tCost(X=%.3f T=%.3f)' % \
            (iter, time.time() - total_time, cost, cv_cost)
    return rbm



# batch_size=20,          # 1-online, n < input_data.shape[0] - batch learning, None or input_data.shape[0] - full batch learning
# idx_data = range(input_data.shape[0])
# np.random.shuffle(idx_data)
# idx_batch = range(0, len(idx_data), batch_size)
        
# for i_in_batch in idx_batch:
# FULL BATCH
# batch_data = input_data[idx_data[i_in_batch:(i_in_batch + batch_size)], :] if not is_sparse \
#              else np.asarray(input_data[idx_data[i_in_batch:(i_in_batch + batch_size)], :].todense())
# v = batch_data



def test_rbm11(learning_rate=0.1, k=1, training_epochs=1000):
    data = np.array([[1,1,1,0,0,0],
                     [1,0,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,1,1,1,0],
                     [0,0,1,1,0,0],
                     [0,0,1,1,1,0]])

    rng = np.random.RandomState(123)
    # construct RBM
    rbm = RBM(input=data, n_visible=6, n_hidden=2, rng=rng)

    # train
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)

        cost = rbm.get_reconstruction_cross_entropy()
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost

    # test
    v = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0]])
    print rbm.reconstruct(v)

if __name__ == "__main__":
    test_rbm()

