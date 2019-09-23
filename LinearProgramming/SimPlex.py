# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:54:56 2019

@author: Lenovo
"""
import numpy as np


def preprocessing(A,b,c,sig,scope = None,opt='max'):
        
    for i in range(len(b)):
        if b[i]<0:
            b[i] = -b[i]
            A[i] = -A[i]
            sig[i] = -sig[i]
    if opt != 'max':
        c = [-i for i in c]
#-------------------------------------------------
    if scope is not None:
        for i in range(len(scope)):
            if scope[i] == 0:

                neg_vec=(-A[:,i]).reshape(len(b),1)
                
                A = np.concatenate((A,neg_vec),axis=1)
            elif scope[i] == -1:
                A[:,i] = -A[:,i]
#-------------------------------------------------
    
    E = np.eye(A.shape[0],dtype = np.float)
    base_index = np.full(A.shape[0],-1)
    uv_loc=[]
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if (E[:,i] == A[:,j]).all():
                uv_loc.append((i,j))
    
    for i,j in uv_loc:
        base_index[i]=j
    
    need = base_index==-1
    
    artificial_var = []
    ori_var = [i for i in range(A.shape[1])]
    
    for i in range(A.shape[0]):
        if sig[i] > 0:
            vec = np.zeros((A.shape[0],1))
            vec[i] = -1.0
            A = np.concatenate((A,vec),axis=1)
            ori_var.append(A.shape[1]-1)
            if need[i]:
                
                
                vec = np.zeros((A.shape[0],1))
                vec[i] = 1.0
                A = np.concatenate((A,vec),axis=1)
                
                base_index[i]=A.shape[1]-1
                artificial_var.append(A.shape[1]-1)

        elif sig[i] < 0:
            vec = np.zeros((A.shape[0],1))
            vec[i] = 1.0
            A = np.concatenate((A,vec),axis=1)
            ori_var.append(A.shape[1]-1)
            
            if need[i]:
                base_index[i]=A.shape[1]-1

        elif sig[i] == 0:
            if need[i]:
                vec = np.zeros((A.shape[0],1))
                vec[i] = 1.0
                A = np.concatenate((A,vec),axis=1)
                base_index[i]=A.shape[1]-1
                artificial_var.append(A.shape[1]-1)

    c = np.hstack((c, np.zeros((A.shape[1] - len(c))))) 
    return A,b,c,base_index,artificial_var,ori_var


def phase_1(A,b,base_index,c,artificial_var):
   
    A = A.squeeze()
    Cb = c[base_index]
    z = np.array([np.sum(Cb*A[:,i]) for i in range(A.shape[1])])
    sigma = c - z
    
    while sigma.max() > 0:
        
            #防止循环
#        max_col = sigma.argmax()
        max_col = (sigma>0).argmax()
        theta = b/A[:,max_col]
        #防止出现0/0情况
        for i in range(A.shape[0]):
            if A[i,max_col]>0:
                theta[i] = b[i]/A[i,max_col]
            else:
                theta[i] = -1

        if (theta>=0).any() and not (theta==np.inf).all():
    
            #更新A b 
            theta[theta<0]=np.inf
    
            #若最小的两个theta值相等，优先换掉人工变量
            mi = theta.min()
            xita = theta == mi
            ll = [x in artificial_var for x in base_index]
            l =xita*ll
            if np.sum(l)==0:
                min_row = theta.argmin()
            else:
                min_row = np.array(l).argmax()
                
            #防止循环
            min_row = theta.argmin()

            #更新A,b
            pivot = A[min_row,max_col]
            b[min_row] /= pivot
            A[min_row] /= pivot
    
            for i in range(A.shape[0]):
                if i != min_row:
                    div = A[i][max_col]
                    b[i] -= b[min_row]*div
                    A[i] -= A[min_row]*div
        
            
            base_index[min_row] = max_col
            Cb[min_row] = c[max_col]

        z = np.array([np.sum(Cb*A[:,i]) for i in range(A.shape[1])])
        sigma = c - z

    for i in range(len(base_index)):
        if base_index[i] in artificial_var:
            if Cb[i] != 0:
                print('No feasible solution')
                return None,None,None

    
    return A,b,base_index


def phase_2(A,b,base_index,c2):
    Cb = c2[base_index]
    
    z = np.array([np.sum(Cb*A[:,i]) for i in range(A.shape[1])])
    sigma = c2 - z

    while sigma.max() > 0:
    
        #sigma>0的那些值中，某个对应的A[:,i]都小于0，则无界解
        p_sigma = sigma>0
        for i in range(len(sigma)):
            if p_sigma[i]:
                if (A[:,i]<=0).all():
                    print('Unbounded solution')
                    return None

        #entering var
#        max_col = sigma.argmax()
        max_col = (sigma>0).argmax()
        theta = np.zeros_like(b,dtype=np.float)
        
        for i in range(A.shape[0]):
            if A[i,max_col]>0:
                theta[i] = b[i]/A[i,max_col]
            else:
                theta[i] = -1
        #leaving var
        theta[theta<0]=np.inf
        min_row = theta.argmin()
        
        #更新A,b
        pivot = A[min_row,max_col]
        b[min_row] /= pivot
        A[min_row] /= pivot
        for i in range(A.shape[0]):
            if i != min_row:
                div = A[i][max_col]
                b[i] -= b[min_row]*div
                A[i] -= A[min_row]*div
    
        base_index[min_row] = max_col
        Cb[min_row] = c2[max_col]

        z = np.array([np.sum(Cb*A[:,i]) for i in range(A.shape[1])])
        sigma = c2 - z
    #    table = np.concatenate((c.reshape(1,A.shape[1]),A,sigma.reshape(1,A.shape[1])),axis = 0)
    #    print(table)
    
    #判断无穷多最优解
    if np.sum(sigma==0)>len(base_index):
        print('Infinite solutions')
        
    opt_solution = np.zeros_like(c2)
    opt_solution[base_index]=b
    
    return opt_solution
        


def simplex(LP):
    
    A,b,c,sig,scope,opt=LP.A,LP.b,LP.c,LP.sig,LP.scope,LP.opt
    #preprocessing
    A_,b,c,base_index,artificial_var,ori_var = preprocessing(A,b,c,sig,scope,opt)
    

    
    if scope is None:
        scope = [1 for _ in range(A_.shape[1])]
        
    if A_ is None:
        return None,None        
    #判断
    if len(artificial_var):
    #phase 1
    
        c1 = np.zeros_like(c,dtype=np.float)
        c1[artificial_var] = -1
        
        A_,b,base_index = phase_1(A_,b,base_index,c1,artificial_var)
        
        #phase 2
        if A_ is None:
            return None,None
        
        
#-----------------------------------------------------
        for i in range(len(base_index)):
            for j in range(len(artificial_var)):
                if(base_index[i]>artificial_var[j]):
                    base_index[i] -= 1
                    
#-----------------------------------------------------        
        
        c2 = c[ori_var]

        A_ = A_[:,[ori_var]]
        A_ = A_.squeeze()
        
        opt_solution = phase_2(A_,b,base_index,c2)
        
        if opt_solution is None and opt=='min':
            return None,-np.inf
        elif opt_solution is None and opt=='max':
            return None,np.inf
        
        
        opt_val = np.sum(opt_solution*c2)
#---------------------------------------------------        
        count = 0
        for i in range(len(scope)):
            if scope[i] == -1:
                opt_solution[i] = -opt_solution[i]
            elif scope[i] == 0:
                opt_solution[i] -= opt_solution[A.shape[1]+count]
                c += 1
        opt_solution = opt_solution[0:A.shape[1]]
#---------------------------------------------------        
        if opt=='min':
            return opt_solution,-opt_val
        else:
            return opt_solution,opt_val
            
    else:
        opt_solution = phase_2(A_,b,base_index,c)
        
        if opt_solution is None:
            return None,np.inf
        
        opt_val = np.sum(opt_solution*c)
        
        
#---------------------------------------------------        
        count = 0
        for i in range(len(scope)):
            if scope[i] == -1:
                opt_solution[i] = -opt_solution[i]
            elif scope[i] == 0:
                opt_solution[i] -= opt_solution[A.shape[1]+count]
                c += 1
        opt_solution = opt_solution[0:A.shape[1]]
#---------------------------------------------------        
                    
        
        if opt=='min':
            return opt_solution,-opt_val
        else:
            return opt_solution,opt_val

#if __name__ == '__main__':
#    
#        
#
#    
#    A=[[1,0,0,0.25,-8,-1,9],[0,1,0,0.5,-12,-0.5,3],[0,0,-1,0,0,-1,0]]
#    b=[0,0,-1]
#    c=[0,0,0,0.75,-20,0.5,-6]
#    sig=[0,0]
#    
#
#
#    
#    A=[[1,1,2,1,3],
#       [2,-1,3,1,1]]
#    b=[4,3]
#    c=[2,3,5,2,3]
#    sig=[1,1]
#    
#    A = [[1,1,1],
#         [-2,1,-1],
#         [0,3,1]]
#    b = [4,1,9]
#    c = [-3,0,1]
#    sig = [-1,1,0]    
#    
#    A=[[2,1],
#       [-3,2],
#       [1,1]]
#    b=[5,3,3]
#    c=[20,15]
#    sig=[1,-1,1]    

