# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 15:38:39 2022

@author: WILL LIU
"""
import numpy as np
import scipy.linalg

def streamingCCA(X,Y,eta1=0.0025,eta2=0.0005,init_l1=0,init_l2=0,iterations = False):
    '''
    Input:
    X,Y: data
    eta1,eta2: learning rate
    init_l1,inie_l2: initial lambda
    iterations: max iteration. If False, run all the points in dataset
    Output:
    a,b: canonical correlation components
    list1,list2: the value of lambda1 and lambda2 during the iterations
    corr_list: correlation during the iterations

    '''

    #initialization
    length,m = X.shape
    l1 = init_l1
    l2 = init_l2
    
    list1 = []
    list2 = []
    corr_list = []

    #random strating value
    a = np.random.randn(m,1)
    a,_ = np.linalg.qr(a, mode='reduced')
    b = np.random.randn(m,1)
    b,_ = np.linalg.qr(b, mode='reduced')

    #if iterations is not set, run all the samples in the dataset
    if iterations==False:
        max_iter = length//100
    else:
        max_iter = iterations//100

    #iterations
    for j in range(max_iter):

        #store the result every 100 iterations
        for i in range(100):
    
            ind = j*100+i
            x = X[ind,:]
            y = Y[ind,:]

            #cov matrices
            c12 = np.outer(x,y)
            c11 = np.outer(x,x)
            c22 = np.outer(y,y)
            
            a += eta1*(c12@b-2*l1*(c11@a))
            b += eta1*(c12.T@a-2*l2*(c22@b))
            l1 += eta2*(a.T@c11@a-1)
            l2 += eta2*(b.T@c22@b-1)

        #store the result and compute the current correlation
        X_s = X@a
        Y_s = Y@b    
        list1.append(float(l1.copy()))
        list2.append(float(l2.copy()))
        corr_list.append(np.corrcoef(X_s.T,Y_s.T)[1,0])

        
    return a,b,list1,list2,corr_list

    
def my_CCA(X,Y,n):
    '''
    Input:
    X,Y: data
    n: the target number of cca components
    Output:
    a_,b_t: theoretical values of cca components
    corr_list: a list of correlations
    '''
    #initialization
    length,m = X.shape
    c12 = X.T@Y
    c11 = X.T@X
    c22 = Y.T@Y
    corr_list = []

    #square roor inverse
    c11_inv_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(c11))
    c22_inv_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(c22))

    R = c11_inv_sqrt@c12@c22_inv_sqrt
    eigenvalue,eigenvector = np.linalg.eig(R.T@R)
    a_t = c11_inv_sqrt@eigenvector
    b_t = c22_inv_sqrt@eigenvector

    #projections
    X_t = X@a_t
    Y_t = Y@b_t
    
    for i in range(n):
        corr_list.append(np.corrcoef(X_t.T,Y_t.T)[n+i,i])
        
    return a_t,b_t,corr_list


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import CCA

    #test on the synthetic dataset
    length = 50000
    l1 = np.random.normal(size=length)
    l2 = np.random.normal(size=length)
    l3 = np.random.normal(size=length)
    
    latents = np.array([l1, l1*0.5, l1*0.25, l2*0.7, l2*0.3, l3*0.5]).T
    X = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5
    Y = latents + np.random.normal(size=6 * length).reshape((length, 6))*0.5

    X = X-X.mean(axis=0)
    Y = Y-Y.mean(axis=0)

    #streaming values
    a,b,list1,list2,corr_list = streamingCCA(X, Y,eta1=0.00001,eta2=0.01,init_l1=0,init_l2=0)

    #theoretical values
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)

    #correlation plot
    plt.plot(np.array(list(range(1,length//100+1)))*100,corr_list,label='streaming method')
    plt.plot([100,length],[np.corrcoef(X_c.T,Y_c.T)[1,0],np.corrcoef(X_c.T,Y_c.T)[1,0]],label='built-in cca function')
    plt.xlabel('iterations')
    plt.ylabel('correlation')
    plt.title('mixed learning rate')
    plt.legend()
    plt.show()

    #lambda plot    
    plt.plot(np.array(list(range(1,length//100+1)))*100,list1,label='$\lambda_1$')
    plt.plot(np.array(list(range(1,length//100+1)))*100,list2,label='$\lambda_2$')
    plt.xlabel('iterations')
    plt.legend()
    plt.title('$\eta_1=0.00001,\eta_2=0.01$')
    plt.show()
