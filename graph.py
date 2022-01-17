# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 20:21:54 2022

@author: WILL LIU
"""

import numpy as np
import matplotlib.pyplot as plt
from gen_data import *
from oja import *
from sklearnpca import *
from errorplot import *
from rotation import *


def graph(n_samples,n_features,num):

    #Generate data
    x = generation(n_samples, n_features)
    xn = normalize(x)
    
    #Built-in PCA
    pca = true_pca(xn,n_features,num)
    
    #Oja's method
    V = init_V(n_features,num)
    oja = oja1(xn, V, eta=0.01)
    
    #rotate
    x_r,r = rotate(x,n_features)
    xn_r = normalize(x_r)
    
    #Built-in PCA
    pca_r = true_pca(xn_r,n_features,num)
    
    #Oja's method
    V = init_V(n_features,num)
    oja_r = oja1(xn_r, V, eta=0.01)
    
    
    # Generate test data
    x,l = generation_with_label(20, n_features)
    xn = normalize(x)
    
    # Pre-rotated un-normalized data --PCA
    xmap = x@pca
    xmap1 = xmap[l==1]
    xmap2 = xmap[l==0]
    
    plt.plot(xmap1[:,0],xmap1[:,1],'.')
    plt.plot(xmap2[:,0],xmap2[:,1],'.')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()
    
    # Pre-rotated un-normalized data --Oja
    xmap_oja = x@oja
    xmap1_oja = xmap_oja[l==1]
    xmap2_oja = xmap_oja[l==0]
    
    plt.plot(xmap1_oja[:,0],xmap1_oja[:,1],'.')
    plt.plot(xmap2_oja[:,0],xmap2_oja[:,1],'.')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()
    
    #Post-rotated un-normalized data  -- Oja
    xmap_r = x@r.T@oja_r
    xmap1_r = xmap_r[l==1]
    xmap2_r = xmap_r[l==0]
    
    plt.plot(xmap1_r[:,0],xmap1_r[:,1],'.')
    plt.plot(xmap2_r[:,0],xmap2_r[:,1],'.')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()
    
    #Post-rotated un-normalized data  -- PCA
    
    xmap_pca_r = x@r.T@pca_r
    xmap1_pca_r = xmap_pca_r[l==1]
    xmap2_pca_r = xmap_pca_r[l==0]
    
    plt.plot(xmap1_pca_r[:,0],xmap1_pca_r[:,1],'.')
    plt.plot(xmap2_pca_r[:,0],xmap2_pca_r[:,1],'.')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()


graph(n_samples=10000,n_features=98,num=2)



