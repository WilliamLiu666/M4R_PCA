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

def generation_with_label(nrow,ncol):
    
    x=datasets.make_moons(n_samples=nrow)
    label = x[1]
    y=np.zeros((nrow,ncol))#zeros 
    x=np.concatenate((x[0],y),axis=1)#extend to higher dimensions
    
    return x,label


def normalize(X):

    z = preprocessing.normalize(X,axis=1)
    
    return z

def standardize(X):
    
    '''
    mean = np.mean(X,axis=1)
    mean = mean.reshape(len(mean),1)
    '''
    mean = np.mean(X,axis=0)
    X = X-mean#centralise
    
    return X

def init_V(ncol,num):
    
    V = np.random.randn(ncol+2,num)
    V,_ = np.linalg.qr(V, mode='reduced')

    return V