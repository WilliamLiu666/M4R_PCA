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

nrow = 1000
ncol = 1
num = 2
step = int(nrow/100)

x = generation(nrow, ncol)


x = normalize(x)
x_r,r = rotate(x,ncol)

print(np.linalg.norm(np.abs(r@true_pca(x,ncol,num))-np.abs(true_pca(x_r,ncol,num))))
pca = true_pca(x,ncol,num)
pca_r = true_pca(x_r,ncol,num)
'''
#x = x-np.mean(x,axis=0)
x = normalize(x)

#Built-in PCA
pca = true_pca(x,ncol,num)

#Oja's method
V = init_V(ncol,num)
oja_list = [V.copy()]

for i in range(100):
    V = oja1(x[step*i:step*i+step], V, eta=0.01)
    oja_list.append(V.copy())

error_plot(oja_list,pca,step)


#rotate
x_r,r = rotate(x,ncol)

#Built-in PCA
pca_r = true_pca(x_r,ncol,num)

#Oja's method
V = init_V(ncol,num)
oja_list_r = [V.copy()]

for i in range(100):
    V = oja1(x_r[step*i:step*i+step], V, eta=0.01)
    oja_list_r.append(V.copy())

error_plot(oja_list_r,pca_r,step)
error_plot(oja_list_r,pca@r.T,step)

'''