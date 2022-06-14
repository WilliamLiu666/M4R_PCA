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



#EMNIST dataset
x = np.load('balanced-MNIST.npy')
x = x/255
x = (x - x.mean(axis=0))
X = x[:,300:335]
Y = x[:,400:435]
length = 112800


#theoretical value
cca = CCA(n_components=1)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)
true_corr = np.corrcoef(X_c.T,Y_c.T)[1,0]

#initialization
init_l1=0
init_l2=0

#choices of learning rate
eta1_list = np.linspace(0.00001,0.1,49)
eta2_list = np.linspace(0.00001,0.1,49)
corr_mat = np.zeros((len(eta1_list),len(eta1_list)))

#iterate over [400,800,1600,3200] iterations
for it in [400,800,1600,3200]:
    for i in range(len(eta1_list)):
        for j in range(len(eta2_list)):

            #carry out SGD CCA
            u,v,list1,list2,corr_list = streamingCCA(X,Y,eta1_list[i],eta2_list[j],init_l1,init_l2,iterations=it)
            corr_mat[i,j] = np.abs(corr_list[-1]-true_corr)

    
    np.save('{} iterations,Snythetic'.format(it),corr_mat)#save the data
    corr_df = pd.DataFrame(data=corr_mat,index=np.round(eta1_list,5), columns=np.round(eta2_list,5))#store as a df so that sns can be used
    fig = sns.heatmap(corr_df,xticklabels=11, yticklabels=11)#heatmap
    fig.set(xlabel='$\eta_1$',ylabel='$\eta_2$',title = '{} iterations'.format(it))
    plt.show()



