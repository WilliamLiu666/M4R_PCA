# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:32:29 2022

@author: WILL LIU
"""
from sklearn import datasets
import numpy as np
from sklearn import preprocessing

def generation(nrow,ncol):
    
    x=datasets.make_moons(n_samples=nrow)[0]
    y=np.zeros((nrow,ncol))#zeros 
    x=np.concatenate((x,y),axis=1)#extend to higher dimensions
    
    return x

def normalize(X):

    z = preprocessing.normalize(X,axis=1)
    return z
    
    '''
    r=(X.max(axis=0) - X.min(axis=0))
    r[r==0]=1
    X_std = (X - X.min(axis=0)) / r
    X_scaled = X_std * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
    return X_scaled

    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_filled = std.copy()
    std_filled[std==0] = 1.
    Xbar = ((X-mu)/std_filled)
    
    return Xbar
    '''
    
    

def init_V(ncol,num):
    
    V = np.random.randn(ncol+2,num)
    V,_ = np.linalg.qr(V, mode='reduced')

    return V