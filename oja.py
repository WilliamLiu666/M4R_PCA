# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:48:20 2022

@author: WILL LIU
"""
import numpy as np

def ojafunc(X, V, eta=0.01):
    '''
    Input:
    X: data
    V: initial value of V
    eta: learning rate
    Output:
    V: estimated principal components
    '''
    #initialization
    t,length = X.shape

    #t step iterations
    for i in range(t):

        #compute cov matrix
        x = X[i,:]
        V += eta*np.outer(x,x)@V #step 1
        
        V,_ = np.linalg.qr(V, mode='reduced') #step 2 and step 3
        
    return V
