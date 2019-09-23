# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:24:16 2019

@author: Dell
"""
import numpy as np
from SimPlex import simplex,preprocessing
from Dual import primal_dual,dualsp
class LinearProgramming():
    '''
        b: RHS
        c: profit
        sig: comparison symbol, use 1,0,-1 to represent >=, =, <=
        opt: a str in {'max','min'}
        
        
        e.g.:
        >>>A = [[2,1],
               [-3,2],
               [1,1]]
        >>>b = [5,3,3]
        >>>c = [20,15]
        >>>sig = [1,-1,1]  
        
        >>>LP = LinearProgramming(A,b,c,sig,'min')
        >>>LP.Simplex()
        
    '''
    
    def __init__(self, A, b, c, sig, scope, opt='max'):
        
        A = np.array(A, np.float)
        b = np.array(b, np.float)
        c = np.array(c, np.float)
        sig = np.array(sig, np.float)
        scope = np.array(scope, np.float)
            
        if A.shape[0] != len(b):
            raise ValueError(r'size of A does not match with b.')
        elif A.shape[0] != len(sig):
            raise ValueError(r'size of A does not match with sig.')
        elif A.shape[1] < len(c):
            raise ValueError(r'size of A does not match with c.')
        elif A.shape[1] != len(scope):
            raise ValueError(r'size of A does not match with x\'scope.')
            
        self.A = A
        self.b = b
        self.c = c
        self.sig = sig
        self.opt = opt
        self.scope=scope
        self.cal = False
        self.ifdual=False
        
    def SimPlex(self):
        
        self.opt_sol, self.opt_val = simplex(self)
        self.cal = True
        
        return self.opt_sol, self.opt_val
    
    def CreatDual(self):
        
        if self.ifdual:
            return
        
        self.dual = LinearProgramming(self.A.T, self.c, self.b, self.scope, -self.sig)
        self.dual.opt = 'min' if self.opt == 'max' else 'max'
        self.dual.dual = self
        
        self.dual.ifdual = True
        self.ifdual = True
            
        return self.dual
        
    def Primal_Dual(self,opt_sol=None):
        
        self.CreatDual()
        if self.cal == True:
           self.dual_sol, self.dual_val = primal_dual(self, self.dual)
           
        else:
            self.cal = True
            self.opt_sol = np.array(opt_sol, np.float)
            self.opt_val = self.c @ self.opt_sol
            self.dual_sol, self.dual_val = primal_dual(self, self.dual)
        
        return self.dual_sol, self.dual_val
        
    def DualSimplex(self):
        
        
        self.opt_sol, self.opt_val = dualsp(self)
        
        return self.opt_sol, self.opt_val
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        