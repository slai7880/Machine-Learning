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
    def __init__(self, nodes, learningRate = 0.01): # nodes do not include the biases
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
        self.randomFactor = 2
        self.nodes = nodes
        self.learningRate = learningRate
        self.W = [] # weights
        for i in range(1, len(nodes)):
            weights = np.random.random((nodes[i], nodes[i - 1])) * self.randomFactor
            
            # weights = np.ones((nodes[i - 1], nodes[i]))
            self.W.append(weights)
        self.B = [] # biases
        for i in range(len(nodes)):
            self.B.append([random.uniform(-self.randomFactor,\
                        self.randomFactor) for j in range(nodes[i])])
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
    
    def forward(self, x):
        """Performs the forward propagation.
        Parameters
        ----------
        x : List
            This is one piece of data.
        Returns
        -------
        activations : List[List[float]]
            Records the value at each node BEFORE being activated. Note
            that this list does not include the values in the first
            layer, so len(activations) = len(outputs) - 1.
        outputs : List[List[float]]
            Records the value at each node AFTER being activated. Note
            that this list includes the values in the first layer, so
            len(activations) = len(outputs) - 1.
        """
        activations = []
        outputs = [x]
        for i in range(len(self.W)):
            # print(outputs[-1])
            w = self.W[i]
            z = np.array([1] * w.shape[0])
            for j in range(len(z)):
                lastOutput = outputs[-1]
                z[j] = np.dot(w[j, :], lastOutput) + self.B[i][j]
            activations.append(z)
            outputs.append(self.G(z))
        return activations, outputs
        
        
    
    def backpropagate(self, labels, activations, outputs):
        delta = [0] * (len(outputs))
        delta[-1] = np.multiply(self.deltaG(activations[-1]), (np.array(outputs[-1]) - np.array(labels)))
        for i in range(len(delta) - 2, 0, -1):
            delta[i] = np.multiply(np.dot(self.W[i].transpose(), delta[i + 1]), self.deltaG(activations[i - 1]))
        deltaW = []
        deltaB = []
        for i in range(len(self.W)):
            wGradient = np.matrix(delta[i + 1]).transpose() * np.matrix(outputs[i])
            bGradient = delta[i + 1]
            deltaW.append(wGradient)
            deltaB.append(bGradient)
        return deltaW, deltaB
    
    def update(self, deltaW, deltaB):
        for i in range(len(self.W)):
            self.W[i] -= self.learningRate * deltaW[i]
        for i in range(len(self.B)):
            self.B[i] -= self.learningRate * deltaB[i]
    
    def fit(self, X, labels, trials = 10):
        if len(X) != len(labels) or len(X) == 0:
            print("Invalid arguments in fit: len(X) = " + str(len(X)) + "  len(labels) = " + str(len(labels)))
            exit()
        self.inputSize = len(X[0])
        
        # made categorized labels
        labelDict = {}
        count = 0
        for label in labels:
            if not label in labelDict:
                labelDict[label] = count
                count += 1
        self.outputSize = len(labelDict)
        labelVectors = []
        for label in labels:
            v = [0] * self.outputSize
            v[labelDict[label]] = 1
            labelVectors.append(v)
        
        inputWeights = np.random.random((self.nodes[0], self.inputSize)) * self.randomFactor
        outputWeights = np.random.random((self.outputSize, self.nodes[-1])) * self.randomFactor
        self.W.insert(0, inputWeights)
        self.W.append(outputWeights)
        self.nodes.insert(0, self.inputSize)
        self.nodes.append(self.outputSize)
        self.B.append([random.uniform(-self.randomFactor, self.randomFactor) for i in range(self.outputSize)])
        for trial in range(trials):
            # print("Trial " + str(trial))
            # print("self.W = " + str(self.W))
            deltaW, deltaB = [], []
            for w in self.W:
                deltaW.append(np.zeros(w.shape))
            for b in self.B:
                deltaB.append([0] * len(b))
            cost = 0
            for i in range(len(X)):
                activations, outputs = self.forward(np.array(X[i]))
                # print("outputs[-1] = " + str(outputs[-1]))
                # print("labelVectors[i] = " + str(labelVectors[i]))
                cost += np.linalg.norm(np.array(outputs[-1]) - np.array(labelVectors[i]), 2) / 2
                wGradients, bGradients = self.backpropagate(labelVectors[i], activations, outputs)
                for j in range(len(deltaW)):
                    deltaW[j] += wGradients[j]
                for j in range(len(deltaB)):
                    deltaB[j] += bGradients[j]
            # print("cost = " + str(cost))
            for i in range(len(deltaW)):
                deltaW[i] /= len(X)
                for j in range(len(deltaB[i])):
                    deltaB[i][j] /= len(X)
            self.update(deltaW, deltaB)
            # print("")
    
    
    def predict(self, X):
        results = [0] * len(X)
        for i in range(len(X)):
            dummy, outputs = self.forward(np.array(X[i]))
            M, MI = -maxsize, -1
            for j in range(len(outputs[-1])):
                if outputs[-1][j] > M:
                    M = outputs[-1][j]
                    MI = j
            results[i] = MI
        return results
             
        
#=========================== End of Section ========================#
#========================== Execution Codes ========================#
