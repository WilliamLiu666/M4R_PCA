# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:19:58 2022

@author: WILL LIU
"""

from cca import *
from geneig import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cross_decomposition import CCA


#synthetic dataset
length = 50000
l1 = np.random.normal(size=length)
l2 = np.random.normal(size=length)
l3 = np.random.normal(size=length)
latents = np.array([l1, l1*0.5, l1*0.25, l2*0.7, l2*0.3, l3*0.5]).T

#add different noise
X = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5
Y = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5

#centralise
X = X-X.mean(axis=0)
Y = Y-Y.mean(axis=0)


#plot the firs 200 points
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


#initialization
n=3
eta1_list = np.linspace(0.00001,0.1,49)
eta2_list = np.linspace(0.00001,0.1,49)
corr_mat = np.zeros((n,len(eta1_list),len(eta1_list)))

#theoretical value
cca = CCA(n_components=n)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)
true_corr = [np.corrcoef(X_c.T,Y_c.T)[n+i,i] for i in range(n)]

#iterate over [400,800,1600,3200] iterations
for it in [400,800,1600,3200]:
    for i in range(len(eta1_list)):
        for j in range(len(eta2_list)):

            #carry out GenOja
            u,v,corr_list = genoja(X,Y,n,eta_1=eta1_list[i],eta_2=eta2_list[j],iterations = it)
            for ind in range(n):
                corr_mat[ind,i,j] = np.abs(corr_list[-1,ind]-true_corr[ind])
                
    #save the data
    np.save('{} iterations,Snythetic,GenOja'.format(it),corr_mat)

    #plot each heatmap
    for ind in range(n):
        corr_df = pd.DataFrame(data=np.log(corr_mat[ind,:,:]),index=np.round(eta1_list,5), columns=np.round(eta2_list,5))
        fig = sns.heatmap(corr_df,xticklabels=11, yticklabels=11)
        fig.set(xlabel='$\eta_1$',ylabel='$\eta_2$',title = 'Log error of {} iterations on the {}th component'.format(it,ind))
        plt.show()

