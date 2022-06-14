# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 17:19:25 2022

@author: WILL LIU
"""
import numpy as np

def genoja(X,Y,n,eta_1=0.0005,eta_2=0.005,iterations = False):
    '''
    Input:
    X,Y: data
    n: number of target components
    eta1,eta2: learning rate
    iterations: max iteration. If False, run all the points in dataset
    Output:
    a,b: canonical correlation components
    corr_list: correlation during the iterations
    '''
    
    #initialization and random starting value
    length,m = X.shape
    np.random.seed(999999999)
    V = np.random.randn(m*2,n)
    V,_ = np.linalg.qr(V, mode='reduced')
    U = np.random.randn(m*2,n)
    U,_ = np.linalg.qr(U, mode='reduced')

    #if iterations is not set, run all the samples in the dataset
    if iterations==False:
        max_iter = length//100
    else:
        max_iter = iterations//100
    corr_mat = np.zeros((max_iter,n))
    
    #iterations
    for j in range(max_iter):

        #store the result every 100 iterations
        for i in range(100):

            #initialize the cov matrices
            ind = j*100+i
            x = X[ind,:]
            y = Y[ind,:]
            c12 = np.outer(x,y)
            c11 = np.outer(x,x)
            c22 = np.outer(y,y)
    
            #initialize A
            A = np.zeros((m*2,m*2))
            A[:m,m:] = c12
            A[m:,:m] = c12.T
            
            #initialize B
            B = np.zeros((m*2,m*2))
            B[0:m,0:m] = c11
            B[m:m*2,m:m*2] = c22
            
            #two-step process
            U -= eta_2*(B@U-A@V)
            V += eta_1*U
            a = V[:m]
            a,r = np.linalg.qr(a, mode='reduced')
            a = a*np.sign(np.diagonal(r))#avoid switching sign in QR
            
            b = V[m:]
            b,r = np.linalg.qr(b, mode='reduced')
            b = b*np.sign(np.diagonal(r))#avoid switching sign in QR
            V[:m] = a
            V[m:] = b
            
        #record the result
        a = V[:m]
        b = V[m:]
        X_s = X@a
        Y_s = Y@b
        for k in range(n):
            corr_mat[j,k] = np.corrcoef(X_s.T,Y_s.T)[n+k,k]

    return a,b,corr_mat


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import CCA

    #test on the sythetic data
    length = 50000
    l1 = np.random.normal(size=length)
    l2 = np.random.normal(size=length)
    l3 = np.random.normal(size=length)

    latents = np.array([l1, l1*0.5, l1*0.25, l2*0.7, l2*0.3, l3*0.5]).T
    X = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5
    Y = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5

    X = X-X.mean(axis=0)
    Y = Y-Y.mean(axis=0)

    #theoretical value
    n=3
    a,b,corr_list = genoja(X,Y,n,eta_1=0.0005,eta_2=0.005)
    cca = CCA(n_components=n)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    
    for i in range(n):
        plt.plot(np.array(list(range(1,length//100+1)))*100,corr_list[:,i],label='streaming')
        plt.plot([100,50000],[np.corrcoef(X_c.T,Y_c.T)[n+i,i],np.corrcoef(X_c.T,Y_c.T)[n+i,i]],label='built-in')
        plt.xlabel('iterations')
        plt.ylabel('correlation')
        plt.legend()

