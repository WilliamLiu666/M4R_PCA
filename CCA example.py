# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:48:38 2022

@author: WILL LIU
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA

n = 500
# 2 latents vars:
l1 = np.random.normal(size=n)
l2 = np.random.normal(size=n)

latents = np.array([l1, l1, l2, l2]).T
X = latents + np.random.normal(size=4 * n).reshape((n, 4))
Y = latents + np.random.normal(size=4 * n).reshape((n, 4))

X_train = X[: n // 2]
Y_train = Y[: n // 2]
X_test = X[n // 2 :]
Y_test = Y[n // 2 :]

X_train = X_train-X_train.mean(axis=0)
Y_train = Y_train-Y_train.mean(axis=0)

print("Corr(X)")
print(np.round(np.corrcoef(X.T), 2))
print("Corr(Y)")
print(np.round(np.corrcoef(Y.T), 2))


cca = CCA(n_components=2)
cca.fit(X_train, Y_train)
X_train_r, Y_train_r = cca.transform(X_train, Y_train)
X_test_r, Y_test_r = cca.transform(X_test, Y_test)

plt.plot(X_train_r[:,0],X_train_r[:,1],'.')
plt.plot(X_test_r[:,0],X_test_r[:,1],'.')
plt.show()

plt.plot(Y_train_r[:,0],Y_train_r[:,1],'.')
plt.plot(Y_test_r[:,0],Y_test_r[:,1],'.')
plt.show()

plt.plot(X_train_r[:,0],Y_train_r[:,0],'.')
plt.plot(X_test_r[:,0],Y_test_r[:,0],'.')
plt.show()

X_train@cca.x_weights_[:,0]