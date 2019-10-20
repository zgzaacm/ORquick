import numpy as np
import copy


def d_nd_zeros(zeros_):
    d_zeros_ = np.array([], dtype=np.int).reshape(0, 2)
    nd_zeros_ = np.array([], dtype=np.int).reshape(0, 2)
    while len(zeros_):

        rep_num = np.zeros(zeros_.shape[0], dtype=np.int)
        arg_nd_zeros = np.zeros(zeros_.shape[0], dtype=np.bool)

        c = 0
        for z in zeros_:
            rep_row = np.sum([z[0] == x for x in zeros_[:, 0]])
            rep_col = np.sum([z[1] == y for y in zeros_[:, 1]])

            rep_num[c] = min(rep_row, rep_col)
            c += 1
        arg_d_zero = np.argmin(rep_num)
        d_zeros_ = np.concatenate((d_zeros_, zeros_[arg_d_zero].reshape(1, 2)), axis=0)

        for i in range(zeros_.shape[0]):
            if i != arg_d_zero:
                if zeros_[i][0] == zeros_[arg_d_zero][0] or zeros_[i][1] == zeros_[arg_d_zero][1]:
                    arg_nd_zeros[i] = True

        nd_zeros_ = np.concatenate((nd_zeros_, zeros_[arg_nd_zeros]), axis=0)

        arg_nd_zeros[arg_d_zero] = True
        zeros_ = zeros_[~arg_nd_zeros]
    return d_zeros_, nd_zeros_


def draw_lines(A, d_zeros, nd_zeros):
    marked_row = [i for i in range(len(A)) if i not in d_zeros[:, 0]]
    marked_col = []
    ex_flag = 1
    while ex_flag:
        ex_flag = 0
        for z in nd_zeros:
            if z[0] in marked_row and z[1] not in marked_col:
                marked_col.append(z[1])
                ex_flag = 1

        for col in marked_col:
            dz = d_zeros[d_zeros[:, 1] == col].squeeze()
            if dz.shape[0] and dz[0] not in marked_row:
                marked_row.append(dz[0])
                ex_flag = 1

    lined_row = [i for i in range(len(A)) if i not in marked_row]
    lined_col = marked_col

    unlined_col = [i for i in range(len(A)) if i not in marked_col]
    unlined_row = marked_row

    min_ = A[unlined_row][:, unlined_col].min()
    A[:, unlined_col] = A[:, unlined_col] - min_
    A[lined_row] += min_

    return A


def norm(A):
    min_row = A.min(axis=1)
    A = A - min_row.reshape(len(min_row), 1)
    A = A - A.min(axis=0)
    return A


def final(A_):
    while True:
        zeros = np.argwhere(A_ == 0)
        d_zeros, nd_zeros = d_nd_zeros(zeros)

        if d_zeros.shape[0] == A_.shape[0]:
            break
        A_ = draw_lines(A_, d_zeros, nd_zeros)
    return d_zeros, nd_zeros


def assign(A):
    """

    :param A: efficiency matrix
    :return:assignment matrix, optional value
    """
    if type(A) is not np.ndarray:
        A_ = np.array(A, np.float)

    A_copy = copy.deepcopy(A_)
    A_ = norm(A_)
    d_zeros, nd_zeros = final(A_)
    A_assigned = np.zeros_like(A_)
    for x, y in d_zeros:
        A_assigned[x, y] = 1

    opt_val = np.sum(A_copy * A_assigned)
    return A_assigned, opt_val


if __name__ == '__main__':
    A_ = [[7, 5, 9, 8, 11],
          [9, 12, 7, 11, 9],
          [8, 5, 4, 6, 9],
          [7, 3, 6, 9, 6],
          [4, 6, 7, 5, 11]]

    A_ = [[4, 8, 7, 15, 12],
          [7, 9, 17, 14, 10],
          [6, 9, 12, 8, 7],
          [6, 7, 14, 6, 10],
          [6, 9, 12, 10, 6]]

    A_assign, opt = assign(A_)
