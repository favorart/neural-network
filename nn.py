from collections import defaultdict
import matplotlib.image as mpimg
import matplotlib.pyplot as pl
import matplotlib as plt
import numpy as np
import random
import copy
import sys
import re
import os

from functions import Cost, Activation


""" +Ранняя остановка на основании ошибки на валидационном множестве
    +Регуляризация, ?момент, ?локальная скорость нейронов
"""

mycost = Cost()
myactiv = Activation(type='hyperbolic')

class neural_network(object):
    """ """
    def __init__(self,
                 vn_layers,
                 v_neurons, # [0] + [Activation()] * (len(vn_layers) - 1)
                 cost=Cost(),
                 learning_rate=0.1,
                 momentum_rate=0.9,
                 regularization_rate=0.1
                ):
        """ 
            ----------------------------------------------------
            n_input  = vn_layers[0]
            n_ilayer = vn_layers[i], i=range(1, len(vn_layers)-1)
            n_output = vn_layers[-1]

            layers-list [5, 10, 10, 5]:
            5 input, 2 hidden layers (10 neurons each), 5 output
            ----------------------------------------------------
            v_neurons = [0, Activation(), Activation(), ...]
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
            W = np.array( np.random.normal(0, 0.1, size=(vn_layers[i], vn_layers[i-1] + 1)) )
            self.weights.append(W)

        self.cost = cost
        self.neurons = v_neurons
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

        # with open('out2.txt', 'w') as f: #  >>f,
        # print self.Z, '\n\n', self.A
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

    def accuracy(self, X, Y):
        """ """
        # Feed forward to obtain the output of the network
        P = self.predict_matrix(X)
        # Calc the accuracy
        accuracy = np.sum( np.argmax(P, axis=1) == np.argmax(Y, axis=1) )
        return float(accuracy) / Y.shape[0]

    def evaluate(self, X, Y):
        """ """
        m = X.shape[0]
        Y = np.array(Y)
        # Feed forward to obtain the output of the network
        P = self.predict_matrix(X)
        # print P.shape, Y.shape
        cost = float( np.sum(self.cost.F(Y,P) + self.regularize(), axis=0) )
        # print P.T

        # Apply the regularization
        return  cost / m 

    def regularize(self, type='L2'):
        """ """
        # def d_L1(x): return (x > 0) * 2 - 1.0  # sign(x) = x / abs(x)
        # def d_L2(x): return (x)

        # # regularization_penalty = rp
        # rp = 0.0 if d_regularization_norm is None else d_regularization_norm(self.W[i])
        # if d_regularization_norm is not None:
        #     rp[:, 0] = 0  # do not penalize bias

        reg = 0.
        weights = copy.deepcopy(self.weights)
        # Goes through the list of matrices
        for i in xrange(1, len(weights)):
            # delete bias
            weights[i] = np.delete(weights[i], 0, axis=1) # 0=slice

            size = ( 1, np.prod(weights[i].shape) )
            if type == 'L1':
                reg += float(np.sum(  np.abs(weights[i].reshape(size)   )) )
            else:
                # square the values, because they can be negative
                reg += float(np.sum(np.power(weights[i].reshape(size), 2)) / 2.)
                # Calc the sum at first by rows, than by columns
        return self.reg_rate * reg

    def backprop(self, X, Y):
        """ X, Y - matrices of doubles

            Calculates the list of matrices 'delta_weights':
                         dim(delta_weights) == dim(weights)
        """
        Y = np.array(Y)
        P = self.predict_matrix(X)
        
        n = self.n_layers
        m = X.shape[0]
        # -----------------------------------------------------------------
        nabla_weights = [0] * (self.n_layers)
        delta_weights = [0] * (self.n_layers)

        prev_delta_weights = [np.zeros_like(W) for W in self.weights ]
        # for i in xrange(m):

        output_layer = True
        # Goes through the all layers
        for l in xrange(n-1, 0, -1):
                
            Z = np.array(self.Z[l])
            if output_layer:
                # output layer
                    
                # print l
                # print "( Z=", Z.shape, '- P=', P.shape, ') * Y=', Y.shape
                # print 'N=', self.neurons[-1].D(Z).shape, "Z=", Z.shape

                # P= A[l]
                deltas = self.cost.D(Y,P) * self.neurons[-1].D(Z)
                output_layer = False
            else:
                # any hidden layer
                    
                # print l+1
                # print self.weights[l+1].shape
                # print self.weights[l].shape
                # print "W=", self.weights[l+1][:,1:].T.shape, "D=", deltas.T.shape
                # print 'N=', self.neurons[l+1].D(Z).shape, "Z=", Z.shape

                deltas = np.dot(self.weights[l+1].T[1:,:], deltas.T)
                deltas = deltas.T * self.neurons[l].D(Z)
                # deltas = deltas[1:] # -bias

            A = self.A[l-1] # layer input data
            # A = np.c_[np.ones(A.shape[0]), A] # already biases

            # print"A=", A.T.shape, "D=", deltas.shape # self.A[l][:,1:].shape, 

            # delta_weights[l] += deltas * self.A[l][:,1:]
            nabla_weights[l] = np.dot(A.T, deltas).T / m

        # with open('out3.txt', 'w') as f:
        #     print >>f, nabla_weights, '\n\n\n'
        # -----------------------------------------------------------------
        # for i in range(1, len(delta_weights)):
        #     delta_weights[i] /= m
        #     delta_weights[i][:,1:] += self.weights[i][:,1:] * (self.regularize() / m)  # regularization

        # update weights
        for i in xrange(1, self.n_layers):
            delta_weights[i] =  self.learning_rate * \
                               (self.momentum_rate * prev_delta_weights[i] + # momentum: add last delta
                                nabla_weights[i])                            # value of gradient
                                # self.reg_rate * self.regularized_weights())  # regularization

            # print "W=", self.weights[i].shape, "D=", delta_weights[i].shape

            self.weights[i] -= delta_weights[i]
            prev_delta_weights[i] = delta_weights[i]
        # -----------------------------------------------------------------
     
    def cross_validation(X_test, Y_test, cv_goal=None):
        # if bias
        # cv_input_data = np.c_[np.ones(X_test.shape[0]), X_test]
        P = self.predict_matrix(X_test, add_bias=False)

        if not cv_goal: cv_goal= self.cost.F

        # regularization_norm = L2
        reg = 0.0 if regularization_norm is None else np.sum(map( lambda m: regularization_norm(m), self.W) )

        cv_cost = np.sum( cv_goal(P, Y_test) + reg ) / X_test.shape[0]
        return cv_cost
     
    def forward_pass(self, X):
        CX, Z, A = X, [], []
    
        for i in range(len(self.W)):
    
            Z.append( np.dot(CX, self.W[i].T) )
            CX = self.f_list[i](Z[i])
            A.append(CX)
    
            if i != (len(self.W) - 1):
                CX = np.c_[ np.ones(CX.shape[0]), CX ]
        return A,Z

    def backward_pass(self, X, Y):
     
        # compute cost
        P = self.predict_matrix(X)

        # # regularization_norm = L2
        # reg  = 0. if regularization_norm is None else np.sum( map(lambda m: regularization_norm(m), self.W) )
        # cost = np.sum( goal(P, Y) + reg ) / X.shape[0]
        #  
        #  
        # if n_iter > n_iter_stop_skip:
        #     if do_cv:
        #         if (1 - stop_threshold) * cv_cost[-1] > min(cv_cost):
        #             break
        #     else:
        #         if (1 - stop_threshold) * cost[-1] > min(cost):
        #             break
        #     if cost[-1] <= min_train_cost:
        #         break
        #     if len(cv_cost) > 0 and cv_cost[-1] <= min_cv_cost:
        #         break
        #     if len(cost) > 1 and abs(cost[-1] - cost[-2]) < tolerance:
        #         break

    