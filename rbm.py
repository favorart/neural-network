import numpy as np
import functions
import sys


"""
    модель
    вход
        размерности видимого и скрытого слоев, 
        k - количество итераций семплирования по Гиббсу. 
        
        датасет первой домашки 
        (в режиме обучения без учителя)
        
        Не забывайте выводить ошибку валидации.
        
        попрошу вас добавить в модель какую нибудь фичу
        типа регуляризации, момента, или отрисовать выученные признаки.
"""
class RBM(object):
    """ Restricted Boltzmann Machine """ 
    def __init__(self,
                 n_visible=2,
                 n_hidden=3,
                 rnd=np.random.RandomState(1234) # random state
                ):
        """ Constructor """
        self.rnd = rnd
        
        self.n_visible = n_visible # num of units in visible (input) layer
        self.n_hidden  = n_hidden  # num of units in hidden layer
        # biases
        self.hbias = self.rnd.normal(0, 0.1, size=n_hidden)  # initialize h bias 0
        self.vbias = self.rnd.normal(0, 0.1, size=n_visible) # initialize v bias 0

        # initialize weights matrix uniformly
        self.W = np.array( self.rnd.normal(0, 0.1, size=(n_visible, n_hidden)) )

    def sample(self, m):
        """
        Sample 0 or 1 with respect to given matrix of probabilities
        :param m: matrix of probabilities
        :return: binary matrix with same shape
        """
        return (np.random.uniform(0, 1, m.shape) < m).astype(float)

    def v_to_h(self, input_data, do_sampling=True):
        """ Compute hidden state
            input_data   - numpy ndarray
            do_sampling  - do binary sample or not
            
            Returns the data representation in the hidden space
        """
        if len(input_data.shape) == 1:
            input_data.shape = (1, input_data.shape[0])
        h = sigmoid(np.dot(np.c_[np.ones(input_data.shape[0]), input_data], np.vstack((self.b, self.W))))
        if do_sampling:
            h = self.sample(h)
        return h

    def h_to_v(self, input_data, do_sampling=False):
        """ Restore input data using hidden space representation
            input_data   - data representation in hidden space
            do_sampling  - do binary sample or not (doesn't matter in gaus-bin mode)
            
            Returns the data representation in the original space
        """
        v = sigmoid(np.dot(np.c_[np.ones(input_data.shape[0]), input_data], np.vstack((self.a, self.W.T))))
        if do_sampling:
            v = self.sample(v)
        return v

    def train(self,
              input_data,
              cd_k = 1,
              learning_rate = 0.1,
              momentum_rate = 0.9,
              max_iter = 10000,
              batch_size=20,
              stop_threshold=0.15,
              goal = euclidian,
              cv_input_data = None,
              regularization_rate = 0.1,
              regularization_norm = None,
              d_regularization_norm = None,
              do_visible_sampling=False,
              n_iter_stop_skip=10,
              min_train_cost = np.finfo(float).eps,
              min_cv_cost = np.finfo(float).eps,
              tolerance=0,
              is_sparse=False,
              verbose=True):
        """
        Train RBM with contrastive divergence algorithm
        :param input_data: numpy ndarray
        :param cd_k: number of iterations in Gibbs sampling
        :param learning_rate: small number
        :param momentum_rate: small number
        :param max_iter: maximum number of iteration
        :param batch_size: 1 - online learning, n < input_data.shape[0] - batch learning,
        None or input_data.shape[0] - full batch learning
        :param stop_threshold: stop training if (1 - stop_threshold)*(current cost) > min(cost);
        is crossvalidation set presented then cv_cost is used
        :param goal: train cost function f: ndarray_NxM -> array_N, N - number of examples;
         in other words it returns cost for each of examples
        :param cv_input_data: numpy ndarry
        :param regularization_rate: small number
        :param regularization_norm: norm to penalize model, if set then it will be included in cost
        :param d_regularization_norm: derivative of norm to penalize model, if set then it will be takken into account
        in error calculation
        :param neural_local_gain: tuple (bonus, penalty, min, max), type of adaptive learning rate;
        https://www.cs.toronto.edu/~hinton/csc321/notes/lec9.pdf
        :param do_visible_sampling: do sampling of visible units in contrastive divergence,
        http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf 3.2 Updating the visible states (page 6)
        :param n_iter_stop_skip: number of iteration skip shecking of stop conditions
        :param min_train_cost: minimum value of cost on train set
        :param min_cv_cost: minimum value of cost on crossvalidation set
        :param tolerance: minimum step of cost
        :param verbose: logging
        :return: values of cost in each of iterations, if cv_cost is set then also returned list of cv_goal values
        """
        cost = []
        cv_cost = []
        last_delta_W = np.zeros(self.W.shape)
        last_delta_a = np.zeros(self.a.shape)
        last_delta_b = np.zeros(self.b.shape)
        for n_iter in range(max_iter):
            t_start = time.clock()
            idx_data = range(input_data.shape[0])
            np.random.shuffle(idx_data)
            idx_batch = range(0, len(idx_data), batch_size)
            for i_in_batch in idx_batch:
                batch_data = input_data[idx_data[i_in_batch:(i_in_batch + batch_size)], :] if not is_sparse \
                    else np.asarray(input_data[idx_data[i_in_batch:(i_in_batch + batch_size)], :].todense())
                v = batch_data
                if self.mode == "pois-bin":
                    pois_N = np.sum(v, axis=1)
                nabla_W = np.zeros(self.W.shape)
                nabla_a = np.zeros(self.a.shape)
                nabla_b = np.zeros(self.b.shape)
                for k in range(cd_k + 1):
                    h = self.compute_output(v, do_sampling=False)
                    if k == cd_k:
                        # accumulate negative phase
                        nabla_W -= np.dot(v.T, h)
                        if self.mode == 'bin-bin':
                            nabla_a -= np.sum(v, axis=0)
                        elif self.mode == 'gaus-bin':
                            nabla_a -= np.sum(np.repeat(self.a, v.shape[0]).reshape((v.shape[0], v.shape[1]), order='F'), axis=0)
                        nabla_b -= np.sum(h, axis=0)
                        break
                    h = self.sample(h)
                    if k == 0:
                        # accumulate positive phase
                        nabla_W += np.dot(v.T, h)
                        if self.mode == 'bin-bin':
                            nabla_a += np.sum(v, axis=0)
                        elif self.mode == 'gaus-bin':
                            nabla_a += np.sum(np.repeat(self.a, v.shape[0]).reshape((v.shape[0], v.shape[1]), order='F'), axis=0)
                        nabla_b += np.sum(h, axis=0)
                    v = self.generate_input(h, do_sampling=False, pois_N=pois_N)
                    if do_visible_sampling:
                        v = self.sample(v)
                nabla_W /= batch_size
                nabla_a /= batch_size
                nabla_b /= batch_size

                # update weights
                regularization_penalty = 0.0 if d_regularization_norm is None else d_regularization_norm(self.W)
                delta_W = learning_rate  * (
                    momentum_rate * last_delta_W +
                    nabla_W -
                    (0.0 if d_regularization_norm is None else regularization_rate * regularization_penalty)
                )
                self.W += delta_W
                last_delta_W = delta_W
                delta_a = learning_rate * (
                    momentum_rate * last_delta_a +
                    nabla_a
                )
                self.a += delta_a
                last_delta_a = delta_a
                delta_b = learning_rate * (
                    momentum_rate * last_delta_b +
                    nabla_b
                )
                self.b += delta_b
                last_delta_b = delta_b

            # compute cost
            if not is_sparse:
                cost.append(np.sum(goal(input_data,
                                        self.generate_input(self.compute_output(input_data, do_sampling=True), do_sampling=True)) +
                                   (0.0 if regularization_norm is None else regularization_norm(self.W))
                ) / input_data.shape[0])
            if cv_input_data is not None:
                cv_cost.append(np.sum(goal(np.asarray(cv_input_data.todense()),
                                    self.generate_input(self.compute_output(np.asarray(cv_input_data.todense()), do_sampling=True), do_sampling=True))
                                      +
                                      (0.0 if regularization_norm is None else regularization_norm(self.W))
                ) / cv_input_data.shape[0])

            t_total = time.clock() - t_start

            if verbose:
                if len(cost) > 0 and len(cv_cost) > 0:
                    print('Iteration: %s (%s s), train/cv cost: %s / %s' % (n_iter, t_total, cost[-1], cv_cost[-1]))
                elif len(cost) > 0:
                    print('Iteration: %s (%s s), train cost = %s' % (n_iter, t_total, cost[-1]))
                elif len(cv_cost) > 0:
                    print('Iteration: %s (%s s), train cv_cost = %s' % (n_iter, t_total, cv_cost[-1]))

            if n_iter > n_iter_stop_skip:
                if len(cv_cost) > 0:
                    if (1 - stop_threshold) * cv_cost[-1] > min(cv_cost):
                        break
                else:
                    if (1 - stop_threshold) * cost[-1] > min(cost):
                        break
                if len(cost) > 0 and cost[-1] <= min_train_cost:
                    break
                if len(cv_cost) > 0 and cv_cost[-1] <= min_cv_cost:
                    break
                if len(cost) > 1 and abs(cost[-1] - cost[-2]) < tolerance:
                    break
            
    def contrastive_divergence(self, X, k=1, lr=0.1):
        """ Contrastive Divergence (CD)-k """
        
        ph_mean, ph_sample = self.sample_h_given_v(self.input)
        chain_start = ph_sample
        for step in xrange(k):
            if step == 0:
                nv_means, nv_samples, \
                nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples, \
                nh_means, nh_samples = self.gibbs_hvh(nh_samples)
        
        # chain_end = nv_samples
        self.W += lr * (np.dot(self.input.T, ph_mean) - np.dot(nv_samples.T, nh_means))
        self.vbias += lr * np.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * np.mean(ph_mean - nh_means, axis=0)
        # cost = self.get_reconstruction_cross_entropy()
        # return cost

    def sample_h_given_v(self, v0_sample):
        """ """
        pre_sigmoid_h1, h1_mean = self.prop_up(v0_sample)
        # discrete: binomial
        h1_sample = self.rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)
        return [h1_mean, h1_sample]
        
    def sample_v_given_h(self, h0_sample):
        """ This function infers state of visible units given hidden units """
        # compute the activation of the visible given the hidden sample
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
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
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


def test_rbm(learning_rate=0.1, k=1, training_epochs=1000):
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
    
