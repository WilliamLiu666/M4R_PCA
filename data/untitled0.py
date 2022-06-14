# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:38:48 2022

@author: WILL LIU
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

eta1_list = np.linspace(0.00001,0.1,49)
eta2_list = np.linspace(0.00001,0.1,49)
for it in [400,800,1600,3200]:

    corr_mat = np.load('{} iterations,Snythetic.npy'.format(it))
    corr_df = pd.DataFrame(data=np.log(np.abs(corr_mat)),index=np.round(eta1_list,5), columns=np.round(eta2_list,5))
    fig = sns.heatmap(corr_df,xticklabels=11, yticklabels=11)
    fig.set(xlabel='$\eta_1$',ylabel='$\eta_2$',title = '{} iterations'.format(it))
    plt.show()