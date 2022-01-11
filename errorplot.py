# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 20:12:36 2022

@author: WILL LIU
"""

import matplotlib.pyplot as plt
import numpy as np

def error_plot(oja_list,true_pca,step):
    
    error_list=[]
    
    for i in oja_list:
        error_list.append(np.linalg.norm(np.abs(i)-np.abs(true_pca)))
        
    plt.plot([step*i for i in range(1,len(error_list)+1)],error_list)
    plt.show()