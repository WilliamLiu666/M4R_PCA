# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:48:20 2022

@author: WILL LIU
"""
import numpy as np

def ojafunc(X, V, eta=0.01):
    
    t,length = X.shape
    
    for i in range(t):
        x = X[i,:]
        V += eta*np.outer(x,x)@V #step 1
        V,_ = np.linalg.qr(V, mode='reduced') #step 2 and step 3
        
    return V

def oja2(X, V, eta, V_true):
    
    t,length = X.shape
    
    for i in range(t):
        x = X[i,:]
        y = np.dot(x,V)
        x = x.reshape([len(x),1])
        V += eta * y*(x- y*V)
        
        error = np.linalg.norm(np.abs(V)-np.abs(V_true)) #Since the sign may change, take the difference between abs value
        
    return V,error