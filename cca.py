# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 15:38:39 2022

@author: WILL LIU
"""
import numpy as np
import scipy.linalg

def streamingCCA(X,Y,eta1=0.0025,eta2=0.0005,init_l1=0,init_l2=0,iterations = False):
    
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
    
    if iterations==False:
        max_iter = length//100
    else:
        max_iter = iterations//100
        
    for j in range(max_iter):
        for i in range(100):
    
            ind = j*100+i
            
            x = X[ind,:]
            y = Y[ind,:]
            
            c12 = np.outer(x,y)
            c11 = np.outer(x,x)
            c22 = np.outer(y,y)
            
            a += eta1*(c12@b-2*l1*(c11@a))
            b += eta1*(c12.T@a-2*l2*(c22@b))
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
    
    length = 50000
    l1 = np.random.normal(size=length)
    l2 = np.random.normal(size=length)
    l3 = np.random.normal(size=length)

    latents = np.array([l1, l1*0.5, l1*0.25, l2*0.7, l2*0.3, l3*0.5]).T
    X = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5
    Y = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5

    X = X-X.mean(axis=0)
    Y = Y-Y.mean(axis=0)
    
    a,b,list1,list2,corr_list = streamingCCA(X, Y,eta1=0.00001,eta2=0.01,init_l1=0,init_l2=0)
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    
    plt.plot(np.array(list(range(1,length//100+1)))*100,corr_list,label='streaming method')
    plt.plot([100,length],[np.corrcoef(X_c.T,Y_c.T)[1,0],np.corrcoef(X_c.T,Y_c.T)[1,0]],label='built-in cca function')
    plt.xlabel('iterations')
    plt.ylabel('correlation')
    plt.title('mixed learning rate')
    plt.legend()
    plt.show()
    
    plt.plot(np.array(list(range(1,length//100+1)))*100,list1,label='$\lambda_1$')
    plt.plot(np.array(list(range(1,length//100+1)))*100,list2,label='$\lambda_2$')
    plt.xlabel('iterations')
    plt.legend()
    plt.title('$\eta_1=0.00001,\eta_2=0.01$')
    plt.show()