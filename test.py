# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:09:20 2022

@author: William
"""


import numpy as np
import matplotlib.pyplot as plt
from gen_data import *
from oja import *
from sklearnpca import *
from errorplot import *
from rotation import *

nrow = 1000
ncol = 10
num = 2

x,l = generation_with_label(nrow, ncol)


xn = normalize(x)
xn_r,r = rotate(xn,ncol)

#print(np.linalg.norm(np.abs(r@true_pca(x,ncol,num))-np.abs(true_pca(xn_r,ncol,num))))
pca = true_pca(xn,ncol,num)
pca_r = true_pca(xn_r,ncol,num)

xmap = x@r.T@pca_r
xmap1 = xmap[l==1]
xmap2 = xmap[l==0]

plt.plot(xmap1[:,0],xmap1[:,1],'.')

plt.plot(xmap2[:,0],xmap2[:,1],'*')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()



xmap = x@pca
xmap1 = xmap[l==1]
xmap2 = xmap[l==0]

plt.plot(xmap1[:,0],xmap1[:,1],'.')

plt.plot(xmap2[:,0],xmap2[:,1],'*')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()



