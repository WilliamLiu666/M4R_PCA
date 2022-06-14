# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:32:54 2022

@author: WILL LIU
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from gen_data import *
from sklearn.decomposition import PCA
from oja import *

x=datasets.make_moons(n_samples=10000)[0]
plt.plot(x[:,0],x[:,1],'.')
plt.title('Two Moons Dataset')
plt.show()

x = np.concatenate((x,np.zeros((10000,100))),axis=1)
xn = centralize(x)
num = 2
it = 1000

pca_obj = PCA(n_components=(102))
pca_obj.fit(x)
pca = pca_obj.components_.T[:,:num]

error_mat = np.zeros((it,num))
var_mat = np.zeros((it,num))
V = init_V(102,num)
var_mat[0,:] = np.diag(V.T@xn.T@xn@V/50000)
error_mat[0,:] = np.abs(var_mat[0,:]-pca_obj.explained_variance_[:num])


for i in range(it):
    print(i)
    V = ojafunc(xn[i*10:(i+1)*10,:], V ,0.005)
    var_mat[i,:] = np.diag(V.T@xn.T@xn@V/10000)
    error_mat[i,:] = np.abs(var_mat[i,:]-pca_obj.explained_variance_[:num])
    
    
color_list=['blue','g','r']    
for i in range(num):
    plt.plot(np.array(range(it))*10,error_mat[:,i],color=color_list[i],label='{}th'.format(i))

plt.title('Error plot')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.show()

for i in range(num):
    plt.plot(np.array(range(it))*10,var_mat[:,i],color=color_list[i],label='{}th streaming'.format(i))
    plt.plot([0,it*10],[pca_obj.explained_variance_[i],pca_obj.explained_variance_[i]],color=color_list[i],linestyle='dashed',label='{}th theoretical'.format(i))

plt.title('Variance plot')
plt.xlabel('Iteration')
plt.ylabel('Variance')
plt.legend()
plt.show()