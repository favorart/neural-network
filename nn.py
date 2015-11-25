import numpy as np
import random
import copy
import sys
import re

from functions import Cost, Activation, Regularization


class NeuralNetwork(object):
    """ """
    def __init__(self,
                 vn_layers,
                 v_neurons,
                 cost=Cost(),
                 learning_rate=0.1,
                 momentum_rate=0.9,
                 # локальная скорость нейронов
                 local_nuruons_rate=0.,
                 # Регуляризация
                 regularization_rate=0.,
                 regularization=Regularization(),
                 rnd=(lambda size, margin=0.1: np.random.normal(0, margin, size))
                ):
        """ 
            ----------------------------------------------------
            n_input  = vn_layers[0]
            n_ilayer = vn_layers[i], i=range(1, len(vn_layers)-1)
            n_output = vn_layers[-1]

            layers-list [5, 10, 10, 5]:
            5 input, 2 hidden layers (10 neurons each), 5 output
            ----------------------------------------------------
            v_neurons = [ Activation() ] * (len(vn_layers) - 1)
            ----------------------------------------------------
        """
        self.layers = vn_layers
        self.n_layers = len(vn_layers)
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.reg_rate = regularization_rate

        self.weights = [0]
        for i in xrange(1, len(self.layers)): # skip zero layer
            # create nxM+1 matrix (+bias!) with random floats in range [-1; 1]
            size=(vn_layers[i], vn_layers[i-1] + 1)
            W = np.array( rnd(size) )
            self.weights.append(W)

        self.cost = cost
        self.neurons = [0] + v_neurons
        self.regular = regularization

        self.Z = None
        self.A = None

    def predict_matrix(self, X):
        """ 
            X - matrix of inputs by rows
            Z - list of matrices of sums(weights x inputs) 
            A - list of matrices of activation functions' values

            A[0] - matrix, that keeps (bias =[1] * m) and input matrix X, and
                           its dimensions are ( rowsX * (1 + colsX) )
            A[1] - matrix, that keeps activation functions' values of first layer and
                           its dimensions are ( rowsX * (1 + self.layers[0]) )

            Returns the matrix of nn's predictions (A[-1]) 
        """
        self.Z = [0]
        self.A = [ copy.deepcopy(X) ]

        m = X.shape[0]
        # For each nn's layer except the input
        for i in xrange(1, self.n_layers):
            # Add the bias column to the previous (activation functions' matrix)
            self.A[i-1] = np.c_[np.ones(m), self.A[i-1]]
            # for all neurons in current layer multiply corresponds neurons
            self.Z.append( np.dot(self.A[i-1], self.weights[i].T) )
            # in previous layers by the appropriate weights and sum the productions
            self.A.append( np.array(self.neurons[i].F(self.Z[i])) )
            # apply activation function for each value
        return self.A[-1]

    def predict(self, input):
        """ """
        if input.shape[1] != 1: input = input.T

        self.Z = [0]
        self.A = [ np.array(input) ] # nx1 vector

        for i in xrange(1, self.n_layers):
            self.A[i-1]  = np.vstack(([1.], self.A[i-1])) # +bias
            self.Z.append( np.dot(self.A[i-1], self.weights[i].T) )
            self.A.append( np.array(self.neurons[i].F(self.Z[i])) )

        return self.A[-1]

    def accuracy(self, X, Y, P=None):
        """ """
        m = X.shape[0]
        Y = np.array(Y)

        if P is None:
            # Feed forward to obtain the output of the network
            P = self.predict_matrix(X)

        # Calculate the accuracy
        accuracy = float(np.sum( np.argmax(P, axis=1) == np.argmax(Y, axis=1) ))
        return accuracy / m

    def calculate_cost(self, X, Y, P=None):
        """ """
        m = X.shape[0]
        Y = np.array(Y)

        if P is None:
            # Feed forward to obtain the output of the network
            P = self.predict_matrix(X)

        # Calculate cost and apply regularization
        cost = (float( np.sum(self.cost.F(Y,P)) )) / m # + self.reg_rate * self.regularize()) / m
        return  cost

    def regularize(self, get_penalty=False):
        """ """
        r, rp = 0., []
        # Goes through the list of matrices
        for i in xrange(1, len(self.weights)):

            W = copy.deepcopy(self.weights[i])
            # delete bias
            W = np.delete(W, 0, axis=1) # 0=slice
            # regularization
            r += float(np.sum( self.regular.F(W) ))

            if get_penalty: # regularization penalty
                rp.append( self.reg_rate * self.regular.D(W) )

        return (rp if get_penalty else r )

    def backprop(self, X, Y, P=None):
        """ X, Y - matrices of doubles

            Calculates the list of matrices 'delta_weights':
                         dim(delta_weights) == dim(weights)
        """
        if  P is None:
            P = self.predict_matrix(X)
        
        n = self.n_layers
        m = X.shape[0]
        Y = np.array(Y)
        # -----------------------------------------------------------------
        nabla_weights      = [0] * (self.n_layers)
        # prev_delta_weights = [ np.zeros_like(W) for W in self.weights ]
        
        output_layer = True
        # Goes through the all layers
        for l in xrange(n-1, 0, -1):
                
            Z = np.array(self.Z[l])
            if output_layer: # output layer
                deltas = self.cost.D(Y,P) * self.neurons[-1].D(Z)
                # print deltas.shape
                output_layer = False
            else: # any hidden layer
                deltas = np.dot(self.weights[l + 1].T[1:,:], deltas.T)   # -bias
                deltas = deltas.T * self.neurons[l + 1].D(Z)

            A = self.A[l-1].T # already biases
            nabla_weights[l] = np.dot(A, deltas).T / m
        # -----------------------------------------------------------------
        # update weights
        for i in xrange(1, n):
            # rp = self.regularize(True)   # regularization rate applied
            # print nabla_weights[i].shape, rp[i].shape

            delta_weights = (self.learning_rate * \
                            # (self.momentum_rate * prev_delta_weights[i] +   # momentum: add last delta
                             nabla_weights[i])                              # value of gradient
                             # + np.c_[ np.ones(rp[i].shape[0]), rp[i] ])   # regularization penalty

            # print "W=", self.weights[i].shape, "D=", delta_weights[i].shape
            self.weights[i] -= delta_weights
            # prev_delta_weights[i] = delta_weights
        # -----------------------------------------------------------------

    def forward_pass(self, X):
        CX, Z, A = X, [], []
    
        for i in range(len(self.W)):
    
            Z.append( np.dot(CX, self.W[i].T) )
            CX = self.f_list[i](Z[i])
            A.append(CX)
    
            if i != (len(self.W) - 1):
                CX = np.c_[ np.ones(CX.shape[0]), CX ]
        return A,Z

    def backprop1(self, X, Y):
                 # learning_rate = 0.1,
                 # momentum_rate = 0.9,
                 # regularization_rate = 0.0001,
                 # regularization = Regularization('L2'),
                 # neural_local_gain = None, # bonus, penalty, min, max

                 # max_iter = 10000,
                 # min_cost = np.finfo(float).eps,
                 # n_iter_stop_skip = 10,
                 # stop_threshold = 0.05,
                 # tolerance = np.finfo(float).eps,
                 # verbose=True):
        m = X.shape[0]
        input_data = np.c_[np.ones(m), X]
        
        if neural_local_gain is not None:
            nlg_bonus, nlg_penalty, nlg_min, nlg_max = neural_local_gain
        
        last_delta_W = [ np.zeros(np.prod(w.shape)).reshape(w.shape) for w in self.W ]

        # neural local gain: learning rate modifier for each of weights in network
        nlg = [ np.ones(w.shape) for w in self.W ] if neural_local_gain is not None else None
        
        # forward pass
        f_z, z = self.forward_pass(X)
        P = f_z[-1]

        # backward pass
        nabla_W = [None] * len(self.W)

        # W = [ W1, W2 ]
        # A = [ X,  A1, P ]
        # Z = [     Z1, Z2]

        out = True
        for i_layer in reversed(range(len(self.W))):
            
            if out: # output layer
                dE_dz = d_goal(Y, P) * (d_f_list[i_layer](z[i_layer]))
                print 'z',i_layer
                out = False
            else: # any hidden layer
                dE_dz = self.W[i_layer + 1].T[1:, :].dot(dE_dz_next.T).T * d_f_list[i_layer](z[i_layer])
                print 'w', i_layer + 1,'z',i_layer

            layer_input_data = None
            if i_layer == 0: # FYI: batch data already contains zero column with ones
                layer_input_data = input_data
                print 'a', 0
            else:
                layer_input_data = np.c_[np.ones(f_z[i_layer - 1].shape[0]), f_z[i_layer - 1]]
                print 'a', i_layer - 1

            nabla_W[i_layer] = np.dot(layer_input_data.T, dE_dz).T / m

        # update weights
        for i_layer in range(len(self.W)):
            regularization_penalty = 0.0 if d_regularization_norm is None else d_regularization_norm(self.W[i_layer])
            if d_regularization_norm is not None:
                regularization_penalty[:, 0] = 0  # do not penalize bias

            delta_W = (learning_rate if nlg is None else learning_rate * nlg[i_layer]) * (
                momentum_rate * last_delta_W[i_layer] +  # momentum: add last delta
                nabla_W[i_layer] +  # value of gradient
                (0.0 if d_regularization_norm is None else regularization_rate * regularization_penalty)  # regularization
            )
            if nlg is not None:
                c = delta_W * last_delta_W[i_layer] >= 0
                nlg[i_layer] = neuron_local_gain_update(nlg[i_layer], c, nlg_bonus, nlg_penalty, nlg_min, nlg_max)
            self.W[i_layer] -= delta_W
            last_delta_W[i_layer] = delta_W
