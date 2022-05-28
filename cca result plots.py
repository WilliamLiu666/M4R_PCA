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



#EMNIST dataset
x = np.load('balanced-MNIST.npy')
x = x/255
x = (x - x.mean(axis=0))
X = x[:,300:335]
Y = x[:,400:435]
length = 112800
'''

#synthetic dataset
length = 50000
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
'''

'''
#streamingCCA method
cca = CCA(n_components=1)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)
true_corr = np.corrcoef(X_c.T,Y_c.T)[1,0]

init_l1=0
init_l2=0

eta1_list = np.linspace(0.00001,0.1,24)
eta2_list = np.linspace(0.00001,0.1,24)
corr_mat = np.zeros((len(eta1_list),len(eta1_list)))

for it in [200,400,800,1600,3200]:
    for i in range(len(eta1_list)):
        print(it,i)
        for j in range(len(eta2_list)):
            u,v,list1,list2,corr_list = streamingCCA(X,Y,eta1_list[i],eta2_list[j],init_l1,init_l2,iterations=it)
            corr_mat[i,j] = np.abs(corr_list[-1]-true_corr)
    
    np.save('{} iterations,Snythetic'.format(it),corr_mat)
    corr_df = pd.DataFrame(data=corr_mat,index=np.round(eta1_list,5), columns=np.round(eta2_list,5))
    fig = sns.heatmap(corr_df,xticklabels=11, yticklabels=11)
    fig.set(xlabel='$\eta_1$',ylabel='$\eta_2$',title = '{} iterations'.format(it))
    plt.show()
'''
'''
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
cca = CCA(n_components=n)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)
true_corr = [np.corrcoef(X_c.T,Y_c.T)[n+i,i] for i in range(n)]
u,v,corr_list = genoja(X,Y,n,eta_1=0.0025,eta_2=0.0025)



for i in range(n):
    plt.plot(np.array(list(range(1,length//100+1)))*100,corr_list[:,i],label='{}th streaming'.format(i+1))
    plt.plot([100,length],[true_corr[i],true_corr[i]],label='{}th theoretical'.format(i+1))
    plt.xlabel('iterations')
    plt.ylabel('correlation')
    plt.title('correlation vs iteration')
    plt.legend()
plt.show()
