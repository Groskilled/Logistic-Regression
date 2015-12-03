#!/usr/bin/python

import csv
import matplotlib.pyplot as plt
import math
import numpy as np

def sigmoid(Theta, X):
    ret = []
    tmp = np.dot(X, Theta)
    for i in range(X.shape[0]):
        ret.append([1.0 / (1.0 + math.exp(-tmp[i]))])
    return (np.array(ret))

def gradientDescent(X, Y):

def main():
    cr = csv.reader(open("../Data.csv","rb"))
    X = []
    Y = []
    for row in cr:
        X.append([float(row[0]), float(row[1])])
        Y.append(float(row[2]))
    X = np.insert(np.array(X, dtype=float), 0, 1.0, axis=1)
    Theta = np.array([[0.0], [0.0], [0.0]])
    print(sigmoid(Theta, X))

if __name__ == "__main__":
    main()

