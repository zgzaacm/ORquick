# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:54:56 2019

@author: Lenovo
"""
import numpy as np
import copy
from BasicFun import Artificial, phase


def simplex(LPP):
    LP = copy.copy(LPP)

    # preprocessing
    LP, base_index, artificial_var, ori_var = Artificial(LP)

    # 判断
    if len(artificial_var):
        # phase 1

        c1 = np.zeros_like(LP.c, dtype=np.float)
        c1[artificial_var] = -1

        LP, base_index, _ = phase('origin', LP, base_index, c1, artificial_var)

        # phase 2
        if base_index is None:
            return None, None

        # -----------------------------------------------------
        for i in range(len(base_index)):
            for j in range(len(artificial_var)):
                if (base_index[i] > artificial_var[j]):
                    base_index[i] -= 1

        # -----------------------------------------------------

        c2 = LP.c[ori_var]

        LP.A = LP.A[:, [ori_var]]
        LP.A = LP.A.squeeze()

        LP, base_index, opt_solution = phase('origin', LP, base_index, c2)

        if opt_solution is None and LP.opt == 'min':
            return None, -np.inf
        elif opt_solution is None and LP.opt == 'max':
            return None, np.inf

        opt_val = np.sum(opt_solution * c2)
        # ---------------------------------------------------
        count = 0
        for i in range(len(LP.scope)):
            if LP.scope[i] == -1:
                opt_solution[i] = -opt_solution[i]
            elif LP.scope[i] == 0:
                opt_solution[i] -= opt_solution[LP.A.shape[1] + count]
                LP.c += 1

        opt_solution = opt_solution[0:LP.A.shape[1]]
        # ---------------------------------------------------
        if LP.opt == 'min':
            return opt_solution, -opt_val
        else:
            return opt_solution, opt_val

    else:
        LP, base_index, opt_solution = phase('origin', LP, base_index, LP.c)

        if opt_solution is None:
            return None, np.inf

        opt_val = np.sum(opt_solution * LP.c)

        # ---------------------------------------------------
        count = 0
        for i in range(len(LP.scope)):
            if LP.scope[i] == -1:
                opt_solution[i] = -opt_solution[i]
            elif LP.scope[i] == 0:
                opt_solution[i] -= opt_solution[LP.A.shape[1] + count]
                LP.c += 1

        opt_solution = opt_solution[0:LP.A.shape[1]]
        # ---------------------------------------------------

        if LP.opt == 'min':
            return opt_solution, -opt_val
        else:
            return opt_solution, opt_val
