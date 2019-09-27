import numpy as np


def b_var_range(llp, key):
    if not llp.cal:
        llp.SimPlex()
    # b = llp.opt_sol[0:llp.A.shape[0]] if llp.opt == 'max' else -llp.opt_sol[0:llp.A.shape[1]]
    b = llp.b_last.copy()

    B_inv = np.linalg.inv(llp.A_init[:, llp.base_index])
    sig = np.zeros_like(b)
    range_ = np.zeros_like(b)
    for i in range(len(llp.b)):
        if B_inv[i, key] != 0:
            range_[i] = -b[i] / B_inv[i, key]
            sig[i] = -1 if B_inv[i, key] < 0 else 1
        else:
            range_[i] = np.nan
            sig[i] = 0

    min_ = range_[sig > 0].max() if len(range_[sig > 0]) != 0 else -np.inf
    max_ = range_[sig < 0].min() if len(range_[sig < 0]) != 0 else np.inf

    return min_, max_


def b_range(llp):
    '''

    :param llp: input an LP obj
    :return: an array of ranges of b that remain the same opt_sol
    '''
    l = []
    for i in range(llp.A.shape[0]):
        l.append(b_var_range(llp, i))

    return np.array(l)
