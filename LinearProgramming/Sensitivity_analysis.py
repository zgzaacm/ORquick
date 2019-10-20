import numpy as np
import matplotlib.pyplot as plt
import copy


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


def b_sensitivity(llp, key, start=0, end=None, split_scale=20, plot=False):
    # TODO 待修改参数
    lp = copy.deepcopy(llp)
    b = lp.b
    if end is None:
        end = 2 * b[key]

    X = np.linspace(start, end, split_scale)
    X = X.squeeze()

    Y = np.zeros((1, len(X)))
    Y = Y.squeeze()

    for i in range(len(X)):
        lp.b[key] = X[i]
        lp.cal = False
        print(lp.SimPlex()[1])
        Y[i] = lp.SimPlex()[1]
    if plot:
        plt.figure()
        plt.title("b[%d]-opt_val" % key)
        plt.xlabel("b[%d]" % key)
        plt.ylabel("opt_val")
        plt.plot(X, Y, '^r-')
    return zip(X, Y)


def c_sensitivity(llp, key, start=0, end=None, split_scale=20, plot=False):
    lp = copy.deepcopy(llp)
    if end is None:
        end = 2 * lp.c[key]
    X = np.linspace(start, end, split_scale)

    Y = np.zeros_like(X)
    Y = Y.squeeze()

    for i in range(len(X)):
        lp.c[key] = X[i]
        lp.cal = False
        Y[i] = lp.SimPlex()[1]

    if plot:
        plt.figure()

        plt.title("c[%d]-opt_val" % key)
        plt.xlabel("c[%d]" % key)
        plt.ylabel("opt_val")
        plt.plot(X, Y, '.b-')

    return zip(X, Y)


def var_2_sensitivity_(llp, var1, key1, var2, key2, start1=0, end1=None, start2=0, end2=None):
    lp = copy.deepcopy(llp)

    if end1 is None:
        if var1 == 'b':
            end1 = 2 * lp.b[key1]
        elif var1 == 'c':
            end1 = 2 * lp.c[key1]

    X = np.linspace(start1, end1, 50)

    if end2 is None:
        if var2 == 'b':
            end2 = 2 * lp.b[key2]
        elif var2 == 'c':
            end2 = 2 * lp.c[key2]
    Y = np.linspace(start2, end2, 50)

    Height = np.zeros((len(Y), len(X)))

    if var1 == 'b' and var2 == 'b':
        for i in range(len(Y)):
            for j in range(len(X)):
                lp.b[key1] = X[i]
                lp.b[key2] = Y[j]
                lp.cal = False
                Height[i, j] = lp.SimPlex()[1]

    if var1 == 'b' and var2 == 'c':
        for i in range(len(Y)):
            for j in range(len(X)):
                lp.b[key1] = X[i]
                lp.c[key2] = Y[j]
                lp.cal = False
                Height[i, j] = lp.SimPlex()[1]

    if var1 == 'c' and var2 == 'b':
        for i in range(len(Y)):
            for j in range(len(X)):
                lp.c[key1] = X[i]
                lp.b[key2] = Y[j]
                lp.cal = False
                Height[i, j] = lp.SimPlex()[1]

    if var1 == 'c' and var2 == 'c':
        for i in range(len(Y)):
            for j in range(len(X)):
                lp.c[key1] = X[i]
                lp.c[key2] = Y[j]
                lp.cal = False
                Height[i, j] = lp.SimPlex()[1]
    plt.figure()
    plt.title("{}[{}],{}[{}]-opt_val contour map".format(var1, key1, var2, key2))
    plt.xlabel('%s[%d]' % (var1, key1))
    plt.ylabel('%s[%d]' % (var2, key2))
    plt.contourf(X, Y, Height, 10, alpha=0.5, cmap=plt.cm.hot)
    # 绘制等高线
    C = plt.contour(X, Y, Height, 10, colors='black')
    # 显示各等高线的数据标签
    plt.clabel(C, inline=True, fontsize=10)

    plt.show()
