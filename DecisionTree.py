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
    
    def fit(self, X, labels):
        self.root = self.buildTree(X, labels, 0)
        
    def predict(self, X):
        results = []
        for i in range(len(X)):
            current = self.root
            while current.label is None:
                if X[i][current.targetIndex] < current.targetValue:
                    current = current.left
                else:
                    current = current.right
            results.append(current.label)
        return results