# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:52:12 2019

@author: Dell
"""
import numpy as np

def preprocessing(LP):
#---------------------------------------------    
    if LP.scope is not None:
        for i in range(len(LP.scope)):
            if LP.scope[i] == 0:

                neg_vec=(-LP.A[:,i]).reshape(len(LP.b),1)
                
                LP.A = np.concatenate((LP.A,neg_vec),axis=1)
            elif LP.scope[i] == -1:
                LP.A[:,i] = -LP.A[:,i]

#---------------------------------------------        

    E = np.eye(LP.A.shape[0],dtype = np.float)
    uv_loc=[]
    
    #加松弛变量
    for i in range(LP.A.shape[0]):
        if LP.sig >0:
            vec = np.zeros((LP.A.shape[0],1))
            vec[i] = -1.0
            LP.A = np.concatenate((LP.A,vec),axis=1)          
        elif LP.sig<0:
            vec = np.zeros((LP.A.shape[0],1))
            vec[i] = 1.0
            LP.A = np.concatenate((LP.A,vec),axis=1)

    
    for i in range(LP.A.shape[0]):
        for j in range(LP.A.shape[1]):
            if (E[:,i] == LP.A[:,j]).all():
                uv_loc.append((i,j))
            elif (E[:,i] == -LP.A[:,j]).all():
                LP.A[i,:] = -LP.A[i,:]
                LP.sig[i] = -LP.sig[i]
                LP.b[i] = -LP.b[i]
                uv_loc.append((i,j))
    base_index = np.array(len(uv_loc),np.float)

    for i,j in uv_loc:
        base_index[i]=j

    return LP,base_index




def standardform(LP):
    #
    for i in range(len(LP.b)):
        if LP.b[i] < 0:
            LP.b[i] = -LP.b[i]
            LP.A[i] = -LP.A[i]
            LP.sig[i] = -LP.sig[i]
    if LP.opt != 'max':
        LP.c = [-i for i in LP.c]
        
    if LP.scope is not None:
        for i in range(len(LP.scope)):
            if LP.scope[i] == 0:

                neg_vec=(-LP.A[:,i]).reshape(len(LP.b), 1)
                
                LP.A = np.concatenate((LP.A, neg_vec), axis=1)
            elif LP.scope[i] == -1:
                LP.A[:,i] = -LP.A[:,i]
    
    return LP

def CheckBasic(LP):
    
    E = np.eye(LP.A.shape[0],dtype = np.float)
    base_index = np.full(LP.A.shape[0],-1)
    uv_loc=[]
    
    for i in range(LP.A.shape[0]):
        for j in range(LP.A.shape[1]):
            if (E[:,i] == LP.A[:,j]).all():
                uv_loc.append((i,j))
    
    for i,j in uv_loc:
        base_index[i]=j
    
    return base_index

def preprocessing(LP):
    
    LP = standardform(LP)
    
    base_index = CheckBasic(LP)
    
    need = base_index==-1
    
    artificial_var = []
    ori_var = [i for i in range(LP.A.shape[1])]
    
    for i in range(LP.A.shape[0]):
        if LP.sig[i] > 0:
            vec = np.zeros((LP.A.shape[0], 1))
            vec[i] = -1.0
            LP.A = np.concatenate((LP.A,vec), axis=1)
            ori_var.append(LP.A.shape[1]-1)
            if need[i]:
                
                
                vec = np.zeros((LP.A.shape[0], 1))
                vec[i] = 1.0
                LP.A = np.concatenate((LP.A,vec), axis=1)
                
                base_index[i] = LP.A.shape[1]-1
                artificial_var.append(LP.A.shape[1]-1)

        elif LP.sig[i] < 0:
            vec = np.zeros((LP.A.shape[0], 1))
            vec[i] = 1.0
            LP.A = np.concatenate((LP.A, vec),axis=1)
            ori_var.append(LP.A.shape[1]-1)
            
            if need[i]:
                base_index[i] = LP.A.shape[1]-1

        elif LP.sig[i] == 0:
            if need[i]:
                vec = np.zeros((LP.A.shape[0],1))
                vec[i] = 1.0
                LP.A = np.concatenate((LP.A,vec), axis=1)
                base_index[i] = LP.A.shape[1]-1
                artificial_var.append(LP.A.shape[1]-1)

    LP.c = np.hstack((LP.c, np.zeros((LP.A.shape[1] - len(LP.c))))) 
    
    return LP,base_index,artificial_var,ori_var


    
def enter_leave(way, sigma, LP):
    
    if way == 'origin':
        max_col = (sigma>0).argmax()
        theta = np.zeros_like(LP.b, dtype=np.float)
        for i in range(LP.A.shape[0]):
            if LP.A[i,max_col]>0:
                theta[i] = LP.b[i] / LP.A[i, max_col]
            else:
                theta[i] = -1
        theta[theta < 0]=np.inf
        min_row = theta.argmin()
        
    else:
        min_row = LP.b.argmin()
        theta = sigma / LP.A[min_row,:]
        max_col = (theta[theta > 0]).argmin()
    
    return max_col, min_row
    
def judge(way, sigma, LP):
    if way == 'origin':
        #sigma>0的那些值中，某个对应的A[:,i]都小于0，则无界解
        p_sigma = sigma>0
        for i in range(len(sigma)):
            if p_sigma[i]:
                if (LP.A[:,i] <= 0).all():
                    print('Unbounded solution')
                    return 1
    
    else:
        #b<0的那些值中，某个对应的A[i,:]都大于0，则无解
        p_b = LP.b<0
        for i in range(len(LP.b)):
            if p_b[i]:
                if min(LP.A[i,:]) >= 0:
                    print('No feasible solution')
                    return 2
    return 0

def phase(way,LP,base_index,c,artificial_var=[]):
    
    if len(artificial_var):
        LP.A = LP.A.squeeze()
        
    Cb = c[base_index]
    
    z = np.array([np.sum(Cb*LP.A[:,i]) for i in range(LP.A.shape[1])])
    sigma = c - z
    
    jud = sigma if way == 'origin' else -LP.b
    
    while max(jud > 0):
        
        #判断无界 无解
        num = judge(way, sigma, LP)
        if num == 1:
            return LP,base_index,None
        elif num == 2:
            return LP,None,None
        
        #entering var and leaving var
        
        max_col, min_row = enter_leave(way, sigma, LP)
        
        #主轴变换
        pivot = LP.A[min_row,max_col]
        LP.b[min_row] /= pivot
        LP.A[min_row] /= pivot
        for i in range(LP.A.shape[0]):
            if i != min_row:
                div = LP.A[i][max_col]
                LP.b[i] -= LP.b[min_row]*div
                LP.A[i] -= LP.A[min_row]*div
    
        base_index[min_row] = max_col
        Cb[min_row] = c[max_col]

        z = np.array([np.sum(Cb*LP.A[:,i]) for i in range(LP.A.shape[1])])
        sigma = c - z
        jud = sigma if way == 'origin' else -LP.b
        
    #判断是否可解 是否无穷多最优解
    if len(artificial_var):
        for i in range(len(base_index)):
            if base_index[i] in artificial_var:
                if Cb[i] != 0:
                    print('No feasible solution')
                    return LP,None,None
                
    else:
        if np.sum(sigma==0)>len(base_index):
            print('Infinite solutions')
            
    opt_solution = np.zeros_like(c)
    opt_solution[base_index]=LP.b
    
    return LP,base_index,opt_solution
        
