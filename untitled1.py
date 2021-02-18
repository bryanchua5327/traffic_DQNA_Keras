# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:55:37 2019

@author: user
"""

import numpy as np
np.random.seed(42)
No_of_Cars = [100,38,80,38] #in an hour LE1-LE4
p1 = np.random.poisson(No_of_Cars[0], 20)
p2 = np.random.poisson(No_of_Cars[1], 20)
p3 = np.random.poisson(No_of_Cars[2], 20)
p4 = np.random.poisson(No_of_Cars[3], 20)
Prob_of_Lane=[p1,p2,p3,p4]