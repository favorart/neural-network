import numpy as np
import random
import sys


from funcs import Cost, Activation, Regularization


class BoltzmannMachine(object):
    """ Restricted Boltzmann Machine """ 
    def __init__(self,
                 n_visible,
                 n_hidden,

                 cost=Cost('square'),# goal = euclidian, # train cost function f: ndarray_NxM -> array_N, N - number of examples;
                 mode='bin-bin',
                 learning_rate=0.1,
                 momentum_rate=0.9,
                 regularization_rate=0.1,
                 regularization=None,      # norm to penalize model, if set then it will be included in cost
                 neural_local_gain=None,   # (bonus, penalty, min, max), type of adaptive learning rate
                 
                 rnd=np.random.RandomState(1234),
                 margin=0.0001):
        """ Constructor """
        self.rnd = rnd # random state
        
        self.n_visible = n_visible # num of units in visible (input) layer
        self.n_hidden  = n_hidden  # num of units in hidden layer
        # biases
        self.hbias = self.rnd.normal(0, 0.1, size=n_hidden)  # initialize h bias 0
        self.vbias = self.rnd.normal(0, 0.1, size=n_visible) # initialize v bias 0
        # initialize weights matrix uniformly
        self.W = np.array( self.rnd.normal(0, margin, size=(n_visible, n_hidden)) )

        self.lr=learning_rate
        self.mr=momentum_rate
        self.rr=regularization_rate

        self.cost=cost
        self.mode=mode
        self.regul=regularization
        self.nlg=neural_local_gain
        
        if  self.nlg: # local gain
            # learning rate modifier for each of weights
            self.nlg_W = np.ones_like(self.W)
            self.nlg_v = np.ones_like(self.vbias)
            self.nlg_h = np.ones_like(self.hbias)

    def sample(self, X):
        """ Sample 0 or 1 with respect to given
            X - matrix of probabilities
            Returns the binary matrix of the same shape
        """
        return (np.random.uniform(0, 1, X.shape) < X).astype(float)

    def v_to_h(self, V, do_sampling=True):
        """ Compute the data representation in the hidden space """
        m = V.shape[0]
        if len(V.shape) == 1: V.shape = (1, m)
        # print V.shape, self.W.shape, self.hbias.shape
        H = Activation.sigmoid_function( np.dot(np.c_[np.ones(m), V], np.vstack((self.hbias, self.W)) ) )
        if do_sampling:
            H = self.sample(H)
        return H

    def h_to_v(self, H, do_sampling=False, pois_N=None):
        """ Restore visual data using hidden space representation.
            Returns the data representation in the original space.
        """
        m = H.shape[0]
        pre_distr = np.dot( np.c_[np.ones(m), H], np.vstack((self.vbias, self.W.T)) )

        if self.mode == 'gaus-bin':
            return pre_distr

        elif self.mode == 'pois-bin':
            e = np.exp(pre_distr)
            if pois_N is None:
                return ( e.T / np.sum(e, axis=1) ).T
            else:
                return ( e.T / ((1. / pois_N) * np.sum(e, axis=1)) ).T

        V = Activation.sigmoid_function(pre_distr)
        if do_sampling:
            V = self.sample(V)
        return V

    def evaluate_cost(self, X, p=None, do_v_sampling=True, do_h_sampling=True):
        """ Returns the cost """
        if  p is None:
            p = self.h_to_v(self.v_to_h(X, do_h_sampling), do_v_sampling)
        return np.sum(self.cost.F(X,p) + self.regularize() ) / X.shape[0]

    def regularize(self, penalty=False):
        """ Returns regulariation cost summand or regulariation penalty """
        return self.rr * ((self.regul.D(self.W)  if penalty else
                           self.regul.F(self.W)) if self.regul else 0.)

    def contrastive_divergence(self, X, cd_k=1, # number of iterations in Gibbs sampling
                               do_visible_sampling=False):
        """ Contrastive Divergence (CD)-k """
        m = X.shape[0]

        prev_delta_W     = np.zeros(self.W.shape)
        prev_delta_vbias = np.zeros(self.vbias.shape)
        prev_delta_hbias = np.zeros(self.hbias.shape)

        pois_N = None
        if self.mode == "pois-bin":
            pois_N = np.sum(X, axis=1)

        nabla_W = np.zeros(self.W.shape)
        nabla_v = np.zeros(self.vbias.shape)
        nabla_h = np.zeros(self.hbias.shape)

        for k in xrange(cd_k + 1):
            h = self.v_to_h(X, do_sampling=False)
            if k == cd_k:
                # accumulate negative phase
                nabla_W -= np.dot(X.T, h)
                if self.mode == 'bin-bin':
                    nabla_v -= np.sum(X, axis=0)
                elif self.mode == 'gaus-bin':
                    nabla_v -= np.sum(np.repeat(self.vbias, X.shape[0]).reshape(X.shape, order='F'), axis=0)
                nabla_h -= np.sum(h, axis=0)
                break

            h = self.sample(h)
            if k == 0:
                # accumulate positive phase
                nabla_W += np.dot(X.T, h)
                if self.mode == 'bin-bin':
                    nabla_v += np.sum(X, axis=0)
                elif self.mode == 'gaus-bin':
                    nabla_v += np.sum(np.repeat(self.vbias, X.shape[0]).reshape(X.shape, order='F'), axis=0)
                nabla_h += np.sum(h, axis=0)

            v = self.h_to_v(h, do_sampling=False, pois_N=pois_N)
            if do_visible_sampling:
                v = self.sample(v)
        # end for

        nabla_W /= m
        nabla_v /= m
        nabla_h /= m

        # update weights
        rp = self.regularize(penalty=True)
        delta_W = (self.lr * self.nlg_W if self.nlg else self.lr ) * \
                  (self.mr * prev_delta_W + nabla_W + rp)
        self.W += delta_W
        prev_delta_W = delta_W

        delta_v = (self.lr * self.nlg_v if self.nlg else self.lr ) * \
                  ( self.mr * prev_delta_vbias + nabla_v )
        self.vbias += delta_v
        prev_delta_vbias = delta_v

        delta_h = (self.lr * self.nlg_h if self.nlg else self.lr ) * \
                  (self.mr * prev_delta_hbias + nabla_h )
        self.hbias += delta_h
        prev_delta_hbias = delta_h

        if  self.nlg:
            bonus, penalty, rmin, rmax = self.nlg
            # Update local rate for W
            condition = ((delta_W * prev_delta_W) >= 0)
            self.nlg_W = self.update_local_gain(self.nlg_W, condition, bonus, penalty, rmin, rmax)

            # Update local rate for v
            condition = ((delta_v * prev_delta_vbias) >= 0)
            self.nlg_v = self.update_local_gain(self.nlg_v, condition, bonus, penalty, rmin, rmax)

            # Update local rate for h
            condition = ((delta_h * prev_delta_hbias) >= 0)
            self.nlg_h = self.update_local_gain(self.nlg_h, condition, bonus, penalty, rmin, rmax)

    @staticmethod
    @np.vectorize
    def update_local_gain(local_gain, condition, bonus, penalty, rmin, rmax):
        """ Update neurons local gain due to condition matrix
                local_gain: by neurons
                 condition: matrix
            Returns the new values of local gain
        """
        return   min(bonus + local_gain, rmax) if condition \
            else max(penalty * local_gain, rmin)

    
