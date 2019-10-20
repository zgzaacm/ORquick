import numpy as np


def d_nd_zeros(zeros):
    d_zeros_ = np.array([], dtype=np.int).reshape(0, 2)
    nd_zeros_ = np.array([], dtype=np.int).reshape(0, 2)
    while len(zeros):
        print(zeros)

        rep_num = np.zeros(zeros.shape[0], dtype=np.int)
        arg_nd_zeros = np.zeros(zeros.shape[0], dtype=np.bool)

        c = 0
        for z in zeros:
            rep_row = np.sum([z[0] == x for x in zeros[:, 0]])
            rep_col = np.sum([z[1] == y for y in zeros[:, 1]])

            rep_num[c] = min(rep_row, rep_col)
            c += 1
        arg_d_zero = np.argmin(rep_num)
        d_zeros_ = np.concatenate((d_zeros_, zeros[arg_d_zero].reshape(1, 2)), axis=0)

        for i in range(zeros.shape[0]):
            if i != arg_d_zero:
                if zeros[i][0] == zeros[arg_d_zero][0] or zeros[i][1] == zeros[arg_d_zero][1]:
                    arg_nd_zeros[i] = True

        nd_zeros_ = np.concatenate((nd_zeros_, zeros[arg_nd_zeros]), axis=0)

        arg_nd_zeros[arg_d_zero] = True
        zeros = zeros[~arg_nd_zeros]
    return d_zeros_, nd_zeros_


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

A = np.array(A_, np.float)

min_row = A.min(axis=1)
A = A - min_row.reshape(len(min_row), 1)
A = A - A.min(axis=0)
print(A)
while True:
    # zeros = np.argwhere(A == 0)
    # print(zeros)
    #
    # d_zeros = zeros[0].reshape(1, 2)
    # nd_zeros = np.array([], dtype=np.int).reshape(0, 2)
    # # -----------------------------------------------------
    # # for zero in zeros[1:]:
    # #     if zero[0] not in d_zeros[:, 0] and zero[1] not in d_zeros[:, 1]:
    # #         d_zeros = np.concatenate((d_zeros, zero.reshape(1, 2)), axis=0)
    # #     else:
    # #         nd_zeros = np.concatenate((nd_zeros, zero.reshape(1, 2)), axis=0)
    # num = np.zeros((1, zeros.shape[0]), dtype=np.int).squeeze()
    #
    # p = 0
    # n = 0
    # while p < zeros.shape[0] - 1:
    #
    #     if zeros[p][0] != zeros[p + 1][0]:
    #         for i in range(p - n, p + 1):
    #             num[i] = n + 1
    #     else:
    #         n += 1
    #     p += 1
    #
    # zeros = zeros[num.argsort()]
    # bool_d_zeros = np.full_like(num, True, dtype=np.bool)
    #
    # for i in range(zeros.shape[0]):
    #     zero = zeros[i]
    #     if bool_d_zeros[i]:
    #         bool_ind = zero[0] in zeros[i + 1:, 0] or zero[1] in zeros[i + 1:, 1]
    #         bool_d_zeros[i + 1:][bool_ind] = False
    #
    #
    # d_zeros = zeros[bool_d_zeros == True]
    # nd_zeros = zeros[bool_d_zeros == False]
    #
    # print(A)
    # print(d_zeros)
    # print(nd_zeros)
    #     # -----------------------------------------------------
    zeros = np.argwhere(A == 0)
    d_zeros, nd_zeros = d_nd_zeros(zeros)

    if d_zeros.shape[0] == A.shape[0]:
        break
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

print(d_zeros)
