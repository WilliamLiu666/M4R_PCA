# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 20:43:13 2022

@author: WILL LIU
"""

from scipy.stats import special_ortho_group

def rotate(x,ncol):
    
    r = special_ortho_group.rvs(ncol+2)
    #x = r@x.T
    #x = x.T
    
    x = x@r.T
    
    return x,r

