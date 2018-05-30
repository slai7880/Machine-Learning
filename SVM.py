"""
This program implements the support vector machines algorithm. It is
not functional however. There is a weird bug when passing the values
to the quadratic programming solver.
"""

import numpy as np
import random
import cvxopt
from sys import exit, maxsize

class SVM:
    """This is an implementation of support vector machines.
    """
    def __init__(self, kernel = None):
        """Initializes the object.
        Parameters
        ----------
        kernel : python function
        Returns
        -------
        None
        """
        self.kernel = kernel
        if kernel is None:
            self.kernel = self.polynomialKernel
    
    def setKernel(self, kernel):
        """Sets the kernel function.
        Parameters
        ----------
        kernel : pythong function
        Returns
        -------
        None
        """
        self.kernel = kernel
    
    #################################################################
    # The following functions will be used by default.
    def polynomialKernel(self, x, y, degree = 2, bias = 1):
        return (x * y.T + bias)**degree
    
    #################################################################
    
    def cvxopt_solve_qp(self, P, q, G = None, h = None, A = None, b = None):
        args = [cvxopt.matrix(P), cvxopt.matrix(q)]
        if G is not None:
            args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
            if A is not None:
                args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        print(args)
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(*args)
        if 'optimal' not in sol['status']:
            return None
        return numpy.array(sol['x']).reshape((P.shape[1],))
    
    def binarize(self, labels):
        """Converts the given label list into a binary version. Quits
        if the given list does not contain exactly two unique labels.
        Parameters
        ----------
        labels : List
        Returns
        -------
        results : List[int]
            Only contains 0s and 1s.
        """
        if len(set(labels)) != 2:
            print("List labels does not contain exactly 2 unique labels.")
            exit()
        labelMap = {}
        count = 0
        for i in labels:
            if not i in labelMap:
                labelMap[i] = count
                count += 1
        result = [0] * len(labels)
        for i in range(len(labels)):
            result[i] = labelMap[labels[i]]
        return result
    
    def fit(self, X, labels):
        X = np.matrix(X)
        labels = self.binarize(labels)
        N = len(labels)
        P = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                P[i, j] = labels[i] * labels[j] * self.kernel(X[i, :], X[j, :], 1)
        q = np.zeros((N, 1)) - 1
        G = np.identity(N) * (-1)
        h = np.zeros((N, 1))
        A = np.matrix(labels)
        b = 0.0
        lagranges = self.cvxopt_solve_qp(P, q, G, h, A.astype(np.double), b)
        print(lagranges)
    
    
    def predict(self, X):
        pass