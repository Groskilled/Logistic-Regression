#!/usr/bin/python

import csv
import matplotlib.pyplot as plt

cr = csv.reader(open("../Data.csv","rb"))
X = []
Y = []
for row in cr:
    X.append([float(row[0]), float(row[1])])
    Y.append(float(row[2]))
for i in range(len(X)):
    if (Y[i] == 0):
        plt.plot(X[i][0], X[i][1],'ro')
    else:
        plt.plot(X[i][0], X[i][1],'bs')
plt.show()
