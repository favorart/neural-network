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
        delta_W = self.lr * (
                  self.mr * prev_delta_W +
                  nabla_W - rp)
        self.W += delta_W
        prev_delta_W = delta_W

        delta_v = self.lr * (
                  self.mr * prev_delta_vbias +
                  nabla_v )
        self.vbias += delta_v
        prev_delta_vbias = delta_v

        delta_h = self.lr * (
                  self.mr * prev_delta_hbias +
                  nabla_h )
        self.hbias += delta_h
        prev_delta_hbias = delta_h



class BoltzmannMachine1(object):        
    def contrastive_divergence(self, X, k=1, lr=0.1):
        """ Contrastive Divergence (CD)-k """
        
        ph_mean, ph_sample = self.h_to_v(self.input)
        nh_samples = ph_sample
        for step in xrange(k):
            nv_means, nv_samples, \
            nh_means, nh_samples = self.gibbs_hvh(nh_samples)
        
        self.W += lr * (np.dot(self.input.T, ph_mean) - np.dot(nv_samples.T, nh_means))
        self.vbias += lr * np.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * np.mean(ph_mean - nh_means, axis=0)

    def h_to_v(self, v0_sample):
        """ """
        pre_sigmoid_h1, h1_mean = self.prop_up(v0_sample)
        h1_sample = self.rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)
        return [h1_mean, h1_sample]
        
    def v_to_h(self, h0_sample):
        """ Compute the activation of the visible given the hidden sample """
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        
        # v1_mean = self.prop_down(h0_sample)
        # discrete: binomial
        v1_sample = self.rng.binomial( size=v1_mean.shape, n=1, p=v1_mean)
        return [v1_mean, v1_sample]

    def prop_up(self, v):
        """ This function propagates the visible units activation
            upwards to the hidden units.

            Note: returns also the pre-sigmoid activation of the layer.
            As it will turn out later, due to how Theano deals with
            optimizations, this symbolic variable will be needed to write
            down a more stable computational graph (see details in the
            reconstruction cost function)
        """
        pre_sigmoid_activation = np.dot(v, self.W) + self.hbias
        return  [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    def prop_down(self, h):

        pre_sigmoid_activation = np.dot(h, self.W.T) + self.vbias
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    # gibbs_vhv  performs a step of Gibbs sampling starting from the visible units. (useful for sampling from the RBM)
    # gibbs_hvh  performs a step of Gibbs sampling starting from the hidden units.  (useful for performing CD and PCD updates)
    # 
    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.v_to_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.h_to_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.h_to_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.v_to_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_reconstruction_cross_entropy(self):
        """ """
        pre_sigmoid_activation_h = np.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)
        cross_entropy = - np.mean(
        np.sum(self.input * np.log(sigmoid_activation_v) +
        (1 - self.input) * np.log(1 - sigmoid_activation_v), axis=1))
        return cross_entropy

    def reconstruct(self, v):
        """ """
        h = sigmoid(np.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(np.dot(h, self.W.T) + self.vbias)
        return reconstructed_v


def nlg():
    """
    neural_local_gain: tuple (bonus, penalty, min, max), type of adaptive learning rate;
    https://www.cs.toronto.edu/~hinton/csc321/notes/lec9.pdf
    do_visible_sampling: do sampling of visible units in contrastive divergence,
    http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf 3.2 Updating the visible states (page 6)
    """
    return


    
