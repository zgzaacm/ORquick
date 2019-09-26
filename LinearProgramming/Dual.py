# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:08:00 2019

@author: Dell
"""
import numpy as np
import copy 
from BasicFun import phase,preprocessing
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
    
def dualsimplex(LPP):
    
    LP = copy.copy(LPP)
    
    LP, base_index = preprocessing(LP)

    if len(base_index) != LP.A.shape[0]:
        print("problem not fit dualsimplex")
        return None, None
    
    Cb = LP.c[base_index]
    z = np.array([np.sum(Cb*LP.A[:,i]) for i in range(LP.A.shape[1])])
    sigma = LP.c - z
    if sigma.max() > 0:
        print("problem not fit dualsimplex")
        return None, None
    
    LP, base_index, opt_solution = phase('dual', LP, base_index, LP.c)
    
    if base_index is None:
        return None,None
    
    opt_solution = opt_solution[0:LP.A.shape[1]]
    
    return opt_solution, opt_solution @ LP.c
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    