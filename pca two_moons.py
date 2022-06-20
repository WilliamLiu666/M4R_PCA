# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:32:54 2022

@author: WILL LIU
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from oja import *
from scipy.stats import special_ortho_group

#load the two moon data
x=datasets.make_moons(n_samples=10000)[0]
plt.plot(x[:,0],x[:,1],'.')
plt.title('Two Moons Dataset')
plt.show()

#rotate the data
x = np.concatenate((x,np.zeros((10000,100))),axis=1)
r = special_ortho_group.rvs(100+2)
x = x@r.T

#centralize the data
xn =  x-np.mean(x,axis=0)
num = 3
it = 1000

#theoretical pca value
pca_obj = PCA(n_components=(102))
pca_obj.fit(x)
pca = pca_obj.components_.T[:,:num]

#initialzation
error_mat = np.zeros((it,num))
var_mat = np.zeros((it,num))
V = np.random.randn(102,num)
V,_ = np.linalg.qr(V, mode='reduced')
var_mat[0,:] = np.diag(V.T@xn.T@xn@V/50000)
error_mat[0,:] = np.abs(var_mat[0,:]-pca_obj.explained_variance_[:num])

#Oja
for i in range(it):
    V = ojafunc(xn[i*10:(i+1)*10,:], V ,0.005)
    var_mat[i,:] = np.diag(V.T@xn.T@xn@V/10000)
    error_mat[i,:] = np.abs(var_mat[i,:]-pca_obj.explained_variance_[:num])
    
#set plot colour
color_list=['blue','g','r']

#error plot
for i in range(num):
    plt.plot(np.array(range(it))*10,error_mat[:,i],color=color_list[i],label='{}th'.format(i+1))
plt.title('Error plot')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.show()

#variance plot
for i in range(num):
    plt.plot(np.array(range(it))*10,var_mat[:,i],color=color_list[i],label='{}th streaming'.format(i+1))
    plt.plot([0,it*10],[pca_obj.explained_variance_[i],pca_obj.explained_variance_[i]],color=color_list[i],linestyle='dashed',label='{}th theoretical'.format(i+1))
plt.title('Variance plot')
plt.xlabel('Iteration')
plt.ylabel('Variance')
plt.legend()
plt.show()
