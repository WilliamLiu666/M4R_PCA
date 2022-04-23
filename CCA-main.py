# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:19:58 2022

@author: WILL LIU
"""

from cca import *
from geneig import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

x = np.load('balanced-MNIST.npy')
x = x/255
x = (x - x.mean(axis=0))
X = x[:,300:335]
Y = x[:,400:435]
length = 112800
'''
length = 50000
l1 = np.random.normal(size=length)
l2 = np.random.normal(size=length)
l3 = np.random.normal(size=length)

latents = np.array([l1, l1*0.5, l1*0.25, l2*0.7, l2*0.3, l3*0.5]).T
X = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5
Y = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5

X = X-X.mean(axis=0)
Y = Y-Y.mean(axis=0)
'''
'''
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

eta1=0.0025
eta2=0.0005
init_l1=0
init_l2=0


a,b,list1,list2,corr_list = streamingCCA(X,Y,eta1,eta2,init_l1,init_l2)
cca = CCA(n_components=1)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)

plt.plot(np.array(list(range(1,length//100+1)))*100,corr_list,label='streaming method')
plt.plot([100,length],[np.corrcoef(X_c.T,Y_c.T)[1,0],np.corrcoef(X_c.T,Y_c.T)[1,0]],label='built-in cca function')
plt.xlabel('iterations')
plt.ylabel('correlation')
plt.legend()
plt.title('correlation vs iteration,$\eta_1$={} $\eta_2$={}'.format(eta1,eta2))
plt.show()

plt.plot(np.array(list(range(1,length//100+1)))*100,list1,label='$\lambda_1$')
plt.plot(np.array(list(range(1,length//100+1)))*100,list2,label='$\lambda_2$')
plt.xlabel('iterations')
plt.legend()
plt.title('$\lambda$ vs iteration,$\eta_1$={} $\eta_2$={}'.format(eta1,eta2))
plt.show()
'''

n=3
for eta1 in [0.0001,0.0005,0.001]:
    for eta2 in [0.001,0.005,0.01]:
        a,b,corr_list = geneig(X,Y,n,eta1,eta2)
        cca = CCA(n_components=n)
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)
        
        for i in range(n):
            plt.plot(np.array(list(range(1,length//100+1-4)))*100,corr_list[:,i],label='{}th streaming'.format(i))
            plt.plot([100,length],[np.corrcoef(X_c.T,Y_c.T)[n+i,i],np.corrcoef(X_c.T,Y_c.T)[n+i,i]],label='{}th theoretical'.format(i))
            plt.xlabel('iterations')
            plt.ylabel('correlation')
            plt.title('correlation vs iteration,$\eta_1$={} $\eta_2$={}'.format(eta1,eta2))
            plt.legend()
        plt.show()
