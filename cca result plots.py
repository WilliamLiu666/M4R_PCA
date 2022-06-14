# -*- coding: utf-8 -*-
"""
Created on Sat May 28 22:49:58 2022

@author: WILL LIU
"""

from cca import *
from geneig import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cross_decomposition import CCA


'''
#EMNIST dataset
x = np.load('balanced-MNIST.npy')
x = x/255
x = (x - x.mean(axis=0))
X = x[:,300:335]
Y = x[:,400:435]
length = 112800

'''
#synthetic dataset
length = 30000
l1 = np.random.normal(size=length)
l2 = np.random.normal(size=length)
l3 = np.random.normal(size=length)

latents = np.array([l1, l1*0.5, l1*0.25, l2*0.7, l2*0.3, l3*0.5]).T
X = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5
Y = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5

X = X-X.mean(axis=0)
Y = Y-Y.mean(axis=0)

plt.plot(X[:200,0],label='X[:200,0]')
plt.plot(Y[:200,0],label='Y[:200,0]')
plt.legend()
plt.title('first 200 points in the first column')
plt.show()

plt.plot(X[:200,1],label='X[:200,1]')
plt.plot(Y[:200,1],label='Y[:200,1]')
plt.legend()
plt.title('first 200 points in the second column')
plt.show()


#streamingCCA method
cca = CCA(n_components=1)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)
true_corr = np.corrcoef(X_c.T,Y_c.T)[1,0]

init_l1=0
init_l2=0

u,v,list1,list2,corr_list = streamingCCA(X,Y,0.0025,0.0025,init_l1,init_l2)



plt.plot(np.array(list(range(1,length//100+1)))*100,corr_list,label='streaming method')
plt.plot([100,length],[np.corrcoef(X_c.T,Y_c.T)[1,0],np.corrcoef(X_c.T,Y_c.T)[1,0]],label='built-in cca function')
plt.xlabel('iterations')
plt.ylabel('correlation')
plt.legend()
plt.title('correlation vs iteration')
plt.show()

plt.plot(np.array(list(range(1,length//100+1)))*100,list1,label='$\lambda_1$')
plt.plot(np.array(list(range(1,length//100+1)))*100,list2,label='$\lambda_2$')
plt.xlabel('iterations')
plt.legend()
plt.title('$\lambda$ vs iteration')
plt.show()


'''
n=3
cca = CCA(n_components=n)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)
true_corr = [np.corrcoef(X_c.T,Y_c.T)[n+i,i] for i in range(n)]
u,v,corr_list = genoja(X,Y,n,eta_1=0.00075,eta_2=0.005)



for i in range(n):
    plt.plot(np.array(list(range(1,length//100+1)))*100,corr_list[:,i],label='{}th streaming'.format(i+1))
    plt.plot([100,length],[true_corr[i],true_corr[i]],label='{}th theoretical'.format(i+1))
    plt.xlabel('iterations')
    plt.ylabel('correlation')
    plt.title('correlation vs iteration')
    plt.legend()
plt.show()
'''