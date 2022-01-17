# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:09:24 2022

@author: WILL LIU
"""

import numpy as np
import matplotlib.pyplot as plt
from gen_data import *
from oja import *
from sklearnpca import *
from errorplot import *
from rotation import *

def compare(n_samples, n_features, num, learning_rate=0.01):
    
    step = int(n_samples/100)
    x = generation(n_samples, n_features)
    x = standardize(x)
    
    #Built-in PCA
    pca = true_pca(x,n_features,num)
    
    #Oja's method
    V = init_V(n_features,num)
    oja_list = [V.copy()]
    
    for i in range(100):
        V = oja1(x[step*i:step*i+step], V, eta=learning_rate)
        oja_list.append(V.copy())
        
    error_plot(oja_list,pca,step)
    
    #rotate
    x_r,r = rotate(x,n_features)
    
    #Built-in PCA
    pca_r = true_pca(x_r,n_features,num)
    
    #Oja's method
    V = init_V(n_features,num)
    oja_list_r = [V.copy()]
    
    for i in range(100):
        V = oja1(x_r[step*i:step*i+step], V, eta=learning_rate)
        oja_list_r.append(V.copy())
        
    error_plot(oja_list_r,pca_r,step)

n_samples = 10000
n_features = 1
num = 2

compare(n_samples,n_features,num)