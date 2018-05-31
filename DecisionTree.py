"""
This program implements the decision tree classifier. Currently
I have only implemented the binary tree.
"""

import numpy as np
import random
from sys import exit, maxsize

class DecisionTreeNode:
    def __init__(self, targetIndex = None, targetValue = None,\
                    left = None, right = None, label = None):
        """Initializes the node object.
        Parameters
        ----------
        targetIndex : int
            The index of the attribute that is associated with this node.
        targetValue : float
            The value of the attribute that is associated with this node.
        left, right : DecisionTreeNode
            References to the child nodes.
        label : int
            The label that represents the node if at leaf.
        """
        self.targetIndex = targetIndex
        self.targetValue = targetValue
        self.left = left
        self.right = right
        self.label = label

class DecisionTree:
    def __init__(self, evaluate = None, maxDepth = 3, minRecords = 0):
        if maxDepth < 0:
            print("Parameter maxDepth must be nonzero.")
            exit()
        self.evaluate = evaluate
        if evaluate is None:
            self.setEvaluationFunction(self.getGiniIndex)
            
        self.maxDepth = int(maxDepth)
        self.minRecords = minRecords
    
    def setEvaluationFunction(self, evaluate):
        self.evaluate = evaluate    
    
    def getGiniIndex(self, groups, classes):
        """Computes the Gini index for a given split.
        Parameters
        ----------
        groups : List[List[float]]
            Each sublist is a group of data labels.
        classes : List[int]
            Stores the numerical label for each class.
        Returns
        -------
        result : float
        """
        M = 0
        for g in groups:
            M += len(g)
        result = 0
        for group in groups:
            if len(group) > 0:
                giniScore = 0
                for i in classes:
                    proportion = group.count(i) / len(group)
                    giniScore += proportion**2
                result += (1 - giniScore) * len(group) / M
        return result
    
    def split(self, data, labels, targetIndex, breakpoint):
        """Splits the data into two groups by checking if the value
        at certain index is smaller or no less than some number.
        Parameters
        ----------
        data : List[List[float]]
        labels : List[int]
        targetIndex : int
        breakpoint : float
        Returns
        -------
        left, right : List[int]
            Each list stores data labels only.
        """
        left = []
        right = []
        for i in range(len(data)):
            if data[i][targetIndex] < breakpoint:
                left.append(i)
            else:
                right.append(i)
        return left, right
    
    def splitData(self, data, labels):
        """Find the information of the best splitted result.
        Parameters
        ----------
        data : List[List[float]]
        labels : List[int]
        Returns
        -------
        
        """
        bestDataIndex = len(data)
        bestTargetIndex = len(data[0])
        bestScore = maxsize
        bestSplit = []
        
        N = len(data[0])
        classes = set(labels)
        for i in range(len(data)):
            for j in range(N):
                left, right = self.split(data, labels, j, data[i][j])
                giniScore = self.evaluate([[labels[i] for i in left], [labels[i] for i in right]], classes)
                if giniScore < bestScore:
                    bestDataIndex = i
                    bestTargetIndex = j
                    bestScore = giniScore
                    bestSplit = [left, right]
        results = {"targetIndex" : bestTargetIndex,\
                "targetValue" : data[bestDataIndex][bestTargetIndex],\
                "splittedDataIndex" : bestSplit}
        return results
    
    def buildTree(self, X, labels, depth):
        """Builds the decision tree by recursively splitting the data
        till some terminating condition(s) is met.
        Parameters
        ----------
        X : List[List[float]]
        labels : List[int]
        depth : int
        Returns
        -------
        current : DecisionTreeNode
        """
        current = None
        if depth == self.maxDepth or len(X) <= self.minRecords or len(set(labels)) == 1:
            current = DecisionTreeNode(label = np.bincount(labels).argmax())
        else:
            args = self.splitData(X, labels)
            current = DecisionTreeNode(args["targetIndex"], args["targetValue"])
            leftX = [X[i] for i in args["splittedDataIndex"][0]]
            rightX = [X[i] for i in args["splittedDataIndex"][1]]
            leftLabels = [labels[i] for i in args["splittedDataIndex"][0]]
            rightLabels = [labels[i] for i in args["splittedDataIndex"][1]]
            current.left = self.buildTree(leftX, leftLabels, depth + 1)
            current.right = self.buildTree(rightX, rightLabels, depth + 1)
        return current
    
    def sample(self, X, labels, percent):
        """Randomly samples a certain proportion of the data and labels.
        Parameters
        ----------
        X : List[List[float]]
        labels : List[int]
        percent : float
        Returns
        -------
        X2 : List[List[float]]
        labels2 : List[int]
        """
        indices = [i for i in range(len(X))]
        random.shuffle(indices)
        bound = int(len(X) * percent)
        X2, labels2 = [], []
        for i in range(bound):
            X2.append(X[indices[i]])
            labels2.append(labels[indices[i]])
        return X2, labels2
    
    def fit(self, X, labels, rootNumber = 1, percent = 0.8):
        if rootNumber % 2 == 0:
            print("Parameter rootNumber must be odd.")
            exit()
        self.roots = []
        self.rootNumber = rootNumber
        for i in range(rootNumber):
            X2, labels2 = self.sample(X, labels, percent)
            self.roots.append(self.buildTree(X, labels, 0))
        
    def predict(self, X):
        results = []
        for i in range(len(X)):
            result = []
            for root in self.roots:
                current = root
                while current.label is None:
                    if X[i][current.targetIndex] < current.targetValue:
                        current = current.left
                    else:
                        current = current.right
                result.append(current.label)
            results.append(result)
        finalResults = []
        for i in range(len(X)):
            votes = []
            for j in range(len(self.roots)):
                votes.append(results[i][j])
            finalResults.append(np.bincount(votes).argmax())
        return finalResults