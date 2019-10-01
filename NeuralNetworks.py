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
    
    def fit(self, X, Y, batch = 1, epochs = 10, regularC = 0, verbose = False):
        assert len(X) != 0 and len(X) == len(Y) and len(X[0]) != 0
        if isinstance(X, list):
            X = np.matrix(X)
        self.NClass = len(set([y for y in Y]))
        Y = self.vectorize(Y) # essential for network output comparison
        
        # add two weight matrices to the list, given input and output dimensions
        self.W.insert(0, np.random.normal(size = (self.nodes[0], X.shape[1] + 1)))
        self.W.append(np.random.normal(size = (self.NClass, self.nodes[-1] + 1)))
        
        for epoch in range(epochs):
            YTrue, YPredict, self.outputs = [], [], []
            for i in range(X.shape[0]):
                YTrue.append(Y[i])
                YPredict.append(self.forward(X[i, :]))
                if i % batch == 0:
                    self.backprog(YTrue, YPredict, regularC)
                    YTrue, YPredict, self.outputs = [], [], []
    
    def predict(self, X):
        YPredict = []
        for i in range(len(X)):
            YPredict.append(X[i])
        return YPredict
        
    def forward(self, x):
        YPredict = x.T
        for i in range(len(self.W)):
            output = self.W[i] * np.vstack((YPredict, np.matrix([[1]])))
            self.outputs.append(output)
            YPredict = self.activate(output)
        y = [0] * self.NClass
        y[np.squeeze(np.array(YPredict)).argmax()] = 1
        return y
        
        
    def backprog(self, YTrue, YPredict, regularC):
        pass
    
    def getLoss(self, YTrue, YPredict):
        pass
        
#=========================== End of Section ========================#
