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

nrow = 10000
ncol = 10
num = 2
step = int(nrow/100)

x = generation(nrow, ncol)

#x = x-np.mean(x,axis=0)
xn = normalize(x)

#Built-in PCA
pca = true_pca(xn,ncol,num)

#Oja's method
V = init_V(ncol,num)
oja_list = [V.copy()]

for i in range(100):
    V = oja1(xn[step*i:step*i+step], V, eta=0.01)
    oja_list.append(V.copy())

error_plot(oja_list,pca,step)


#rotate
xn_r,r = rotate(xn,ncol)

#Built-in PCA
pca_r = true_pca(xn_r,ncol,num)

#Oja's method
V = init_V(ncol,num)
oja_list_r = [V.copy()]

for i in range(100):
    V = oja1(xn_r[step*i:step*i+step], V, eta=0.01)
    oja_list_r.append(V.copy())

error_plot(oja_list_r,pca_r,step)
#error_plot(oja_list_r,pca@r.T,step)


x,l = generation_with_label(10, ncol)
xn = normalize(x)

xmap = x@pca
xmap1 = xmap[l==1]
xmap2 = xmap[l==0]

plt.plot(xmap1[:,0],xmap1[:,1],'.')
plt.plot(xmap2[:,0],xmap2[:,1],'*')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()


xmap = x@r.T@V
xmap1 = xmap[l==1]
xmap2 = xmap[l==0]

plt.plot(xmap1[:,0],xmap1[:,1],'.')
plt.plot(xmap2[:,0],xmap2[:,1],'*')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()

xmap = xn@r.T@V
xmap1 = xmap[l==1]
xmap2 = xmap[l==0]

plt.plot(xmap1[:,0],xmap1[:,1],'.')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()
plt.plot(xmap2[:,0],xmap2[:,1],'*')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()




xmap = x@r.T@pca_r
xmap1 = xmap[l==1]
xmap2 = xmap[l==0]

plt.plot(xmap1[:,0],xmap1[:,1],'.')
plt.plot(xmap2[:,0],xmap2[:,1],'*')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()


xmap = xn@r.T@pca_r
xmap1 = xmap[l==1]
xmap2 = xmap[l==0]

plt.plot(xmap1[:,0],xmap1[:,1],'.')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()

plt.plot(xmap2[:,0],xmap2[:,1],'*')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()








