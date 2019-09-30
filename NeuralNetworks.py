"""
This program implements the good-old-fashioned neural networks. At
this point there is no special layer and the traditional backpropogation
is used.
"""

import numpy as np
import random
from sys import exit, maxsize

class NeuralNetworks:
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
        self.randomFactor = 2
        self.nodes = nodes
        self.W = [] # weights
        for i in range(1, len(nodes)):
            weights = np.random.random((nodes[i], nodes[i - 1] + 1)) * self.randomFactor
            self.W.append(weights)
        # set the default activation function
        self.setActivationFunction(self.sigmoid, self.dSigmoid)
    
    def setActivationFunction(self, G, deltaG):
        """Sets the avtivation function and its corresponding
        derivative.
        Parameters
        ----------
        G, deltaG : pythong function
        Returns
        -------
        None
        """
        self.G = G
        self.deltaG = deltaG
    
    #################################################################
    # The following functions will be used by default.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def dSigmoid(self, x):
        return np.multiply(self.sigmoid(x), 1 - self.sigmoid(x))
    
    #################################################################
    
    def fit(self, X, Y, batch = 1, epochs = 10, regularC = 0):
        assert len(X) != 0 and len(X) == len(Y) and len(X[0]) != 0
        self.NClass = len(set([y for y in Y]))
        self.W.insert(np.random.random((self.nodes[0], len(X[0]) + 1)) * self.randomFactor, 0)
        self.W.append(np.random.random((self.NClass, len(self.nodes[-1]) + 1)) * self.randomFactor)
        for epoch in range(epochs):
            YTrue, YPredict = [], []
            for i in range(len(X)):
                YTrue.append(Y[i])
                YPredict.append(self.forward(np.array([x for x in X[i]] + [1])))
                if i % batch == 0:
                    self.backprog(YTrue, YPredict, regularC)
                    YTrue, YPredict = [], []
    
    def predict(self, X):
        YPredict = []
        for i in range(len(X)):
            YPredict.append(X[i])
        return YPredict
        
    def forward(self, x):
        pass
        
    def backprog(self, YTrue, YPredict, regularC):
        pass
             
        
#=========================== End of Section ========================#
