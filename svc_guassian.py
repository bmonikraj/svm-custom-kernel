# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 01:10:25 2018

@author: MONIK RAJ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC
d = loadmat("D:/STUDY - IMPORTANT/COMPUTER/Coursera-Machine Learning -By Andrew Ng/XII. Support Vector Machines (Week 7)/ex6/ex6data3.mat")
X = d['X']
Xval = d['Xval']
y = d['y']
yval = d['yval']

def gaussianKernel(U,V,sigma):
        return np.exp((-1/2*sigma*sigma)*(np.sum(np.power((U-V),2))))

'''
gram = np.zeros((len(X), len(X)))
for i in range(0,len(X)):
    for j in range(0,len(X)):
        gram[i][j] = gaussianKernel(X[i],X[j],0.3)
'''

def gram(U,V,sigma=0.1):
    G = np.zeros((U.shape[0], V.shape[0]))
    for i in range(0,U.shape[0]):
        for j in range(0,V.shape[0]):
            G[i][j] = gaussianKernel(U[i],V[j],sigma)
    return G

'''
def wrapper(sigma):
    def gaussianKernel(U,V):
        return np.exp((-1/2*sigma*sigma)*(np.sum(np.power((U-V),2))))
    return gaussianKernel
'''

#clf = SVC(kernel = wrapper(sigma=0.3))
#clf.fit(X,y.reshape(len(y),))
C = s = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

ACC = []

for i in range(0,8):
    a = []
    for j in range(0,8):
        clf = SVC(C = C[i], kernel = "precomputed")
        clf.fit(gram(X,X,s[j]),y.reshape(len(y),))
        yTest = clf.predict(np.dot(Xval,X.T))
        e = yTest - yval.reshape(len(yval))
        acc = e.shape[0] - np.count_nonzero(e)
        acc = (acc*100)/e.shape[0]
        a.append(acc)
    ACC.append(a)

plt.plot(s,ACC[0],s,ACC[1],s,ACC[2],s,ACC[3],s,ACC[4],s,ACC[5],s,ACC[6],s,ACC[7])
