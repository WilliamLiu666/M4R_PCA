# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 15:38:39 2022

@author: WILL LIU
"""
import numpy as np
import scipy.linalg

def streamingCCA(X,Y,n):
    
    length,m = X.shape
    
    a = np.random.randn(m,n)
    a,_ = np.linalg.qr(a, mode='reduced')
    b = np.random.randn(m,n)
    b,_ = np.linalg.qr(b, mode='reduced')
    
    corr_list = np.zeros((n,1000))
    
    eta = 0.00025
    for j in range(0,1000):
        for i in range(100):
    
            ind = j*100+i
            x = X[ind,:]
            y = Y[ind,:]
    
            c12 = np.outer(x,y)
    
            a += eta*c12@b
            a,ar = np.linalg.qr(a, mode='reduced')
            
            for k in range(n):
                a[:,k] = a[:,k]/np.sqrt(a[:,k].T@X[:ind+1,:].T@X[:ind+1,:]@a[:,k])
            a = a*np.sign(np.diagonal(ar))
    
            b += eta*c12.T@a
            b,br = np.linalg.qr(b, mode='reduced')
            for k in range(n):
                b[:,k] = b[:,k]/np.sqrt(b[:,k].T@Y[:ind+1,:].T@Y[:ind+1,:]@b[:,k])
            b = b*np.sign(np.diagonal(br))
    
        X_s = X@a
        Y_s = Y@b    
        for k in range(n):
            corr_list[k,j] = np.corrcoef(X_s.T,Y_s.T)[n+k,k]
        
    return a,b,corr_list
    
def CCA(X,Y,n):
    
    length,m = X.shape
    
    c12 = X.T@Y
    c11 = X.T@X
    c22 = Y.T@Y
    
    
    eigenvalue,eigenvector = np.linalg.eig(c11)
    c11_inv_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(c11))
    
    eigenvalue,eigenvector = np.linalg.eig(c22)
    c22_inv_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(c22))
    
    R = c11_inv_sqrt@c12@c22_inv_sqrt
    eigenvalue,eigenvector = np.linalg.eig(R.T@R)
    
    a_t = c11_inv_sqrt@eigenvector[:,:3]
    b_t = c22_inv_sqrt@eigenvector[:,:3]
    
    X_t = X@a_t
    Y_t = Y@b_t
    
    corr_list = []
    
    for i in range(n):
        corr_list.append(np.corrcoef(X_t.T,Y_t.T)[n+i,i])
        
    return a_t,b_t,corr_list


if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    l1 = np.random.normal(size=100000)
    l2 = np.random.normal(size=100000)
    l3 = np.random.normal(size=100000)
    
    latents = np.array([l1, l1*0.5, l1*0.25, l2*0.7, l2*0.3, l3*0.5]).T
    X = latents + np.random.normal(size=6 * 100000).reshape((100000, 6))*0.5
    Y = latents + np.random.normal(size=6 * 100000).reshape((100000, 6))*0.5
    
    X = X-X.mean(axis=0)
    Y = Y-Y.mean(axis=0)
    
    _,_,corr_list = streamingCCA(X, Y, n=3)
    _,_,my_cca_corr = CCA(X,Y,n=3)
    
    for i in range(3):
        plt.plot(np.array(list(range(1,1001)))*100,corr_list[i,:],label='streaming method {}'.format(i))
        plt.plot([1000,100000],[my_cca_corr[i],my_cca_corr[i]],label='my cca function {}'.format(i))
        plt.xlabel('iterations')
        plt.ylabel('correlation')
        plt.legend()
        plt.show()