"""
This program implements the good-old-fashioned neural networks. At
this point there is no special layer and the traditional backpropogation
is used.
"""

import numpy as np
import random
from sys import exit, maxsize

class NeuralNetwork:
    """This is an implementation of neural networks.
    """
    def __init__(self, nodes): # nodes do not include the biases
        """Initializes the object.
        Parameters
        ----------
        nodes : List[int]
            The number of nodes in each layer. The bias term is NOT
            included in this list.
        learningRate : float
        Returns
        -------
        None
        """
        if len(nodes) < 1:
            print("Not enough layer, len(nodes) = " + str(nodes))
            exit()
        for i in range(len(nodes)):
            if nodes[i] == 0:
                print("Layer " + str(i) + " is empty.")
        self.randomFactor = 1
        self.nodes = nodes
        self.W = [] # weights
        for i in range(1, len(nodes)):
            weights = np.random.normal(size = (nodes[i], nodes[i - 1] + 1))
            self.W.append(weights)
        # set the default activation function
        self.setActivationFunction(self.sigmoid, self.dSigmoid)
    
    def setActivationFunction(self, activate, dActivate):
        """Sets the avtivation function and its corresponding
        derivative.
        Parameters
        ----------
        G, deltaG : pythong function
        Returns
        -------
        None
        """
        self.activate = activate
        self.dActivate = dActivate
    
    #################################################################
    # The following functions will be used by default.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def dSigmoid(self, x):
        return np.multiply(self.sigmoid(x), 1 - self.sigmoid(x))
    
    #################################################################
    
    def vectorize(self, Y):
        results = []
        for i in range(len(Y)):
            y = [0] * self.NClass
            y[Y[i]] = 1
            results.append(y)
        return results
    
    def fit(self, X, Y, batch = 1, epochs = 10, learningRate = 0.001, regularC = 0, verbose = False):
        assert len(X) != 0 and len(X) == len(Y) and len(X[0]) != 0
        if isinstance(X, list):
            X = np.matrix(X)
        self.NClass = len(set([y for y in Y]))
        Y = self.vectorize(Y) # essential for network output comparison
        
        # add two weight matrices to the list, given input and output dimensions
        self.W.insert(0, np.random.normal(size = (self.nodes[0], X.shape[1] + 1)))
        self.W.append(np.random.normal(size = (self.NClass, self.nodes[-1] + 1)))
        for epoch in range(epochs):
            YTrue, YPredict = [], []
            loss = 0
            for i in range(X.shape[0]):
                YTrue.append(Y[i])
                self.outputs = [self.activate(X[i, :].T)]
                YPredict.append(self.forward(X[i, :]))
                loss += self.getLoss(YTrue, YPredict)
                if i % batch == 0:
                    self.backprog(YTrue, YPredict, learningRate, regularC)
                    YTrue, YPredict = [], []
            if verbose:
                print("Epoch: " + str(epoch) + "  loss = " + str(loss))
    
    def predict(self, X):
        if isinstance(X, list):
            X = np.matrix(X)
        outputs = []
        for i in range(X.shape[0]):
            outputs.append(np.squeeze(np.array(self.forward(X[i, :]))))
        outputs = np.array(outputs)
        return outputs.argmax(axis = 1)
    
    def forward(self, x):
        output = x.T
        for i in range(len(self.W)): # this loop is problematic
            A = self.W[i] * np.vstack((self.activate(output), np.matrix([[1]])))
            self.outputs.append(A)
            if i < len(self.W) - 1:
                output = self.activate(A)
            else:
                temp = np.exp(output - np.max(A)) # softmax
                output = temp / temp.sum()
        return output
        
        
    def backprog(self, YTrue, YPredict, learningRate, regularC):
        gradientsBatch = [np.zeros(w.shape) for w in self.W]
        for y in YTrue:
            delta = [self.outputs[-1] - np.matrix(y).T]
            for i in range(len(self.outputs) - 2, 0, -1):
                delta.insert(0, np.multiply(self.dActivate(self.outputs[i]), self.W[i][:, :-1].T * delta[0]))
            gradients = []
            for i in range(len(self.W)):
                gradientsBatch[i] += delta[i] * np.hstack((self.outputs[i].T, np.matrix([[1]])))
        for i in range(len(self.W)):
            self.W[i] -= learningRate * gradientsBatch[i]
            
    
    def getLoss(self, YTrue, YPredict):
        """
        Computes cross entropy loss.
        """
        loss = 0
        for i in range(len(YTrue)):
            loss -= np.dot(YTrue[i], np.log(YPredict[i]))
        return loss
            
        
#=========================== End of Section ========================#
