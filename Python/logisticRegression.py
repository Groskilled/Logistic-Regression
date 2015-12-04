#!/usr/bin/python

import csv
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.optimize as opt
from scipy.optimize import fmin

def sigmoid(X):
    return (1 / (1 + np.exp(-X)))

def cost(Theta, X, Y):
    tmp = sigmoid(X * Theta)
    return (-Y * np.log(tmp) - (1 - Y) * np.log(1 - tmp)).mean()

def gradient(Theta, X, Y):
    hyp = sigmoid(X * Theta) - Y
    grad = Theta
    for i in range(X.shape[1]):
      grad[i] = (hyp * X[:,i].reshape(len(Y), 1).T).mean()
    return grad

def main():
    cr = csv.reader(open("../Data.csv","rb"))
    X = []
    Y = []
    for row in cr:
        X.append([float(row[0]), float(row[1])])
        Y.append([float(row[2])])
    X = np.matrix(X)
    Y = np.matrix(Y)
    Theta = np.matrix(np.zeros(X.shape[1]))
    print(cost(Theta, X, Y))
    print(gradient(Theta, X, Y))

if __name__ == "__main__":
    main()

