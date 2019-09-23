# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:08:00 2019

@author: Dell
"""
import numpy as np
from scipy.linalg import solve

def primal_dual(prim,dual):
    
    index = np.arange(0, len(prim.opt_sol.T))
    row_de = index[prim.opt_sol == 0]
    col_de = index[(prim.A @ prim.opt_sol.T - prim.b) != 0]
    
    Arg = np.hstack((dual.A, dual.b[:,np.newaxis]))
    Arg = np.delete(Arg, row_de, axis=0)
    Arg = np.delete(Arg, col_de, axis=1) 
    sol = solve(Arg[:,:-1], Arg[:,-1])
    
    y = np.ones(dual.A.shape[0])
    y[col_de] = 0
    y[y == 1] = sol
    
    return y, dual.c @ y
    
def dualsp(lp):
    pass
    
    