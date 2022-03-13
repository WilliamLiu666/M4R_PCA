# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:13:18 2022

@author: WILL LIU
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gen_data import *
from oja import *
from sklearnpca import *
from errorplot import *
from rotation import *

'''
x = tf.keras.datasets.cifar10.load_data()[0][0]
x = x.reshape(-1,32*32*3)
x = x/255
x = centralize(x)
'''

from keras.datasets import mnist
x = mnist.load_data()[0][0]
x = x.reshape(-1,28*28)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)



pca = PCA(n_components=(28*28))
pca.X_pca = pca.fit_transform(x)
pca_true = pca.components_[:,:4]


num = 4
n_features = 28*28
V = init_V(n_features-2,num)
oja = oja1(x[:1000,:], V)
print(np.linalg.norm(np.abs(pca_true[:,:16])-np.abs(oja)))
print(oja[:5,:5])

oja = oja1(x[1000:2000,:], oja)
print(np.linalg.norm(np.abs(pca_true[:,:16])-np.abs(oja)))
print(oja[:5,:5])

oja = oja1(x[1000:2000,:], oja)
print(np.linalg.norm(np.abs(pca_true[:,:16])-np.abs(oja)))
print(oja[:5,:5])

oja = oja1(x[1000:2000,:], oja)
print(np.linalg.norm(np.abs(pca_true[:,:16])-np.abs(oja)))
print(oja[:5,:5])

oja = oja1(x[1000:2000,:], oja)
print(np.linalg.norm(np.abs(pca_true[:,:16])-np.abs(oja)))
print(oja[:5,:5])

oja = oja1(x[1000:2000,:], oja)
print(np.linalg.norm(np.abs(pca_true[:,:16])-np.abs(oja)))
print(oja[:5,:5])

oja = oja1(x[1000:2000,:], oja)
print(np.linalg.norm(np.abs(pca_true[:,:16])-np.abs(oja)))
print(oja[:5,:5])


    