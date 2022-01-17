# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:56:35 2022

@author: WILL LIU
"""
from sklearn.decomposition import PCA
import time

def true_pca(x,ncol,num):
    pca_start = time.time()
    pca = PCA(n_components=(ncol+2), svd_solver='full')
    pca.fit(x)
    pca_end = time.time()
    result = pca.components_.T
    #print('Time taken for PCA',pca_end-pca_start)
    #print('varaince ratio:')
    #print(pca.explained_variance_ratio_[:5])
    
    return result[:,:num]