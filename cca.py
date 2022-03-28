# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 15:38:39 2022

@author: WILL LIU
"""
import numpy as np
import scipy.linalg
import scipy.sparse

def streamingCCA(X,Y,n):
    
    length,m = X.shape
    
    a = np.random.randn(m,n)
    a,_ = np.linalg.qr(a, mode='reduced')
    b = np.random.randn(m,n)
    b,_ = np.linalg.qr(b, mode='reduced')
    
    corr_list = np.zeros((n,100))
    
    covx = np.zeros((m,m))
    covy = np.zeros((m,m))
    
    eta = 0.0025
    for j in range(0,30):
        print(j)
        for i in range(500):
    
            ind = j*100+i
            x = X[ind,:]
            y = Y[ind,:]
    
            c12 = np.outer(x,y)
            
            covx = (covx*ind+np.outer(x,x))/(ind+1)
            covy = (covy*ind+np.outer(y,y))/(ind+1)
            
            a += eta*c12@b
            a,ar = np.linalg.qr(a, mode='reduced')            
            #for k in range(n):
                #a[:,k] = a[:,k]/np.sqrt(a[:,k].T@covx@a[:,k])
            a = a*np.sign(np.diagonal(ar))
    
            b += eta*c12.T@a
            b,br = np.linalg.qr(b, mode='reduced')
            #for k in range(n):
                #b[:,k] = b[:,k]/np.sqrt(b[:,k].T@covy@b[:,k])
            b = b*np.sign(np.diagonal(br))
    
        X_s = X@a
        Y_s = Y@b    
        for k in range(n):
            corr_list[k,j] = np.corrcoef(X_s.T,Y_s.T)[n+k,k]
    
    eta = 0.0025
    for j in range(30,100):
        print(j)
        for i in range(500):
    
            ind = j*100+i
            x = X[ind,:]
            y = Y[ind,:]
    
            c12 = np.outer(x,y)
    
            a += eta*c12@b
            a,ar = np.linalg.qr(a, mode='reduced')
            
            #for k in range(n):
                #a[:,k] = a[:,k]/np.sqrt(a[:,k].T@covx@a[:,k])
            a = a*np.sign(np.diagonal(ar))
    
            b += eta*c12.T@a
            b,br = np.linalg.qr(b, mode='reduced')
            #for k in range(n):
                #b[:,k] = b[:,k]/np.sqrt(b[:,k].T@covy@b[:,k])
            b = b*np.sign(np.diagonal(br))
    
        X_s = X@a
        Y_s = Y@b    
        for k in range(n):
            corr_list[k,j] = np.corrcoef(X_s.T,Y_s.T)[n+k,k]
        
    return a,b,corr_list
    
def my_CCA(X,Y,n):
    
    length,m = X.shape
    
    c12 = X.T@Y
    c11 = X.T@X
    c22 = Y.T@Y
    
    
    #eigenvalue,eigenvector = np.linalg.eig(c11)
    c11_inv_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(c11))

    #eigenvalue,eigenvector = np.linalg.eig(c22)
    c22_inv_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(c22))

    R = c11_inv_sqrt@c12@c22_inv_sqrt
    eigenvalue,eigenvector = np.linalg.eig(R.T@R)
    print(eigenvalue)
    a_t = c11_inv_sqrt@eigenvector
    b_t = c22_inv_sqrt@eigenvector
    
    X_t = X@a_t
    Y_t = Y@b_t
    
    corr_list = []
    
    for i in range(n):
        corr_list.append(np.corrcoef(X_t.T,Y_t.T)[n+i,i])
        
    return a_t,b_t,corr_list


if __name__=='__main__':
    '''
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import CCA
    
    l1 = np.random.normal(size=100000)
    l2 = np.random.normal(size=100000)
    l3 = np.random.normal(size=100000)
    
    latents = np.array([l1, l1*0.5, l1*0.25, l2*0.7, l2*0.3, l3*0.5]).T
    X = latents + np.random.normal(size=6 * 100000).reshape((100000, 6))*0.5
    Y = latents + np.random.normal(size=6 * 100000).reshape((100000, 6))*0.5
    
    X = X-X.mean(axis=0)
    Y = Y-Y.mean(axis=0)
    '''
    from sklearn.cross_decomposition import CCA
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    x = tf.keras.datasets.cifar10.load_data()[0][0]
    x = x.reshape(-1,32*32*3)
    x = x/255
    x = (x - x.mean(axis=0))
    X = x[:,1526:1531]
    Y = x[:,2536:2541]
    #'''
    a,b,corr_list = streamingCCA(X, Y, n=5)
    cca = CCA(n_components=5,max_iter=60000)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    
    for i in range(5):
        plt.plot(np.array(list(range(1,101)))*500,corr_list[i,:],label='streaming method {}'.format(i))
        plt.plot([100,50000],[np.corrcoef(X_c.T,Y_c.T)[5+i,i],np.corrcoef(X_c.T,Y_c.T)[5+i,i]],label='built-in cca function {}'.format(i))
        plt.xlabel('iterations')
        plt.ylabel('correlation')
        plt.legend()
        plt.show()
