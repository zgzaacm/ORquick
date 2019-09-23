# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:24:16 2019

@author: Dell
"""
import numpy as np
from SimPlex import simplex 
from Dual import primal_dual
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
    
    def __init__(self, A, b, c, sig, scope=None, opt='max'):
        
        A = np.array(A, np.float)
        b = np.array(b, np.float)
        c = np.array(c, np.float)
        sig = np.array(sig, np.float)
#        scope = np.array(scope,np.float)
            
        if A.shape[0] != len(b):
            raise ValueError(r'size of A does not match with b.')
        elif A.shape[0] != len(sig):
            raise ValueError(r'size of A does not match with sig.')
        elif A.shape[1] < len(c):
            raise ValueError(r'size of A does not match with c.')
        elif A.shape[1] != len(sig):
            raise ValueError(r'size of A does not match with x\'scope.')
            
        self.A = A
        self.b = b
        self.c = c
        self.sig = sig
        self.opt = opt
#        self.scope=scope
        self.cal = False
        
    def SimPlex(self):
        
        self.opt_sol, self.opt_val = simplex(self)
        self.cal = True
        
        return self.opt_sol, self.opt_val
    
    def CreatDual(self):
        
        
        if self.opt == 'max':
            self.dual = LinearProgramming(self.A.T, self.c, self.b, -self.sig, opt='min')
        else:
            self.dual = LinearProgramming(self.A.T, self.c, self.b, -self.sig, opt='max')
            
        return self.Dual
        
    def Primal_Dual(self,opt_sol):
        
        if self.cal == True:
           self.dual_sol, self.dual_val = primal_dual(self, self.dual)
           
        else:
            self.cal = True
            self.opt_sol = np.array(opt_sol, np.float)
            self.opt_val = self.c @ self.opt_sol
            self.dual_sol, self.dual_val = primal_dual(self, self.dual)
        
        return self.dual_sol, self.dual_val
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        