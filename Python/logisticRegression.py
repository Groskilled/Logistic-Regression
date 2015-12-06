#!/usr/bin/python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy import c_, mat, e

def sigmoid(z):
    g = 1. / (1 + e**(-z.A))
    return g

def cost(Theta, X, Y):
    tmp = sigmoid(X.dot(c_[Theta]))
    left = -Y.T.dot(np.log(tmp))
    right = (1 - Y.T).dot(np.log(1 - tmp))
    ret = 1. / X.shape[0] * (left - right)
    return ret[0][0]

def main():
    data = np.loadtxt("ex2data1.txt", delimiter=',')
    X = mat(c_[data[:, :2]])
    Y = c_[data[:, 2]]
    m , n = X.shape
    X = c_[np.ones(m), X]
    Theta = np.zeros(n + 1)
    ret, grad = cost(Theta, X, Y), None
    print "For Theta = \n", Theta, "\ncost is : \n", ret
    print grad
    options = {'full_output': True, 'maxiter': 400}
    theta, ret, _, _, _ = optimize.fmin(lambda t: cost(t, X, Y), Theta, **options)

if __name__ == "__main__":
    main()

