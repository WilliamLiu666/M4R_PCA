# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:44:22 2022

@author: WILL LIU
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from oja import *
from gen_data import *
from sklearnpca import *
from errorplot import *


x = tf.keras.datasets.cifar10.load_data()[0][0]
x = x.reshape(-1,32*32*3)
x = x/255
n_features = 32*32*3
num = 3
xn = centralize(x)
it = 1000

#Built-in PCA
pca_obj = PCA(n_components=(n_features))
pca_obj.fit(x)
pca = pca_obj.components_.T[:,:num]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(list(range(1,11)),pca_obj.explained_variance_ratio_[:10]*100)
plt.plot(list(range(1,11)),[sum(pca_obj.explained_variance_ratio_[:i+1])*100 for i in range(10)])
plt.ylabel('percentage variance ratio')
plt.xlabel('i th eigenvalue')
plt.legend(['eigen value ratio','cumulative eigen value ratio'])
plt.show()

error_mat = np.zeros((it,num))
var_mat = np.zeros((it,num))
V = init_V(n_features,num)
var_mat[0,:] = np.diag(V.T@xn.T@xn@V/50000)
error_mat[0,:] = np.abs(var_mat[0,:]-pca_obj.explained_variance_[:num])


for i in range(it):
    print(i)
    V = ojafunc(xn[i*10:(i+1)*10,:], V ,0.0001)
    var_mat[i,:] = np.diag(V.T@xn.T@xn@V/50000)
    error_mat[i,:] = np.abs(var_mat[i,:]-pca_obj.explained_variance_[:num])
    
    
for i in range(num):
    plt.plot(np.array(range(it))*10,error_mat[:,i],label='{}th'.format(i))

plt.title('Error plot')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.show()


for i in range(num):
    plt.plot(np.array(range(it))*10,var_mat[:,i],label='{}th streaming'.format(i))
    plt.plot([0,it*10],[pca_obj.explained_variance_[i],pca_obj.explained_variance_[i]],label='{}th theoretical'.format(i))

plt.title('Variance plot')
plt.xlabel('Iteration')
plt.ylabel('Variance')
plt.legend()
plt.show()





