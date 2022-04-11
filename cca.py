# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 15:38:39 2022

@author: WILL LIU
"""
import numpy as np
import scipy.linalg

def streamingCCA(X,Y,eta1=0.0025,eta2=0.0005,init_l1=0,init_l2=0):
    
    length,m = X.shape
    
    a = np.random.randn(m,1)
    a,_ = np.linalg.qr(a, mode='reduced')
    b = np.random.randn(m,1)
    b,_ = np.linalg.qr(b, mode='reduced')

    l1 = init_l1
    l2 = init_l2
    
    list1 = []
    list2 = []
    corr_list = []
        
    for j in range(1128):
        for i in range(100):
    
            ind = j*100+i
            
            x = X[ind,:]
            y = Y[ind,:]
            
            c12 = np.outer(x,y)
            c11 = np.outer(x,x)
            c22 = np.outer(y,y)
            
            a += eta1*(c12@b-l1*(c11@a))
            b += eta1*(c12.T@a-l2*(c22@b))
            l1 += eta2*(a.T@c11@a-1)
            l2 += eta2*(b.T@c22@b-1)
    
        X_s = X@a
        Y_s = Y@b    
        list1.append(float(l1.copy()))
        list2.append(float(l2.copy()))
        
        corr_list.append(np.corrcoef(X_s.T,Y_s.T)[1,0])

        
    return a,b,list1,list2,corr_list
    
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
    a_t = c11_inv_sqrt@eigenvector
    b_t = c22_inv_sqrt@eigenvector
    
    X_t = X@a_t
    Y_t = Y@b_t
    
    corr_list = []
    
    for i in range(n):
        corr_list.append(np.corrcoef(X_t.T,Y_t.T)[n+i,i])
        
    return a_t,b_t,corr_list


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import CCA
    
    x = np.load('balanced-MNIST.npy')
    x = x/255
    x = (x - x.mean(axis=0))
    X = x[:,300:335]
    Y = x[:,400:435]
    
    a,b,list1,list2,corr_list = streamingCCA(X, Y)
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    
    plt.plot(np.array(list(range(1,1129)))*100,corr_list,label='streaming method')
    plt.plot([100,112800],[np.corrcoef(X_c.T,Y_c.T)[1,0],np.corrcoef(X_c.T,Y_c.T)[1,0]],label='built-in cca function')
    plt.xlabel('iterations')
    plt.ylabel('correlation')
    plt.legend()
    plt.show()