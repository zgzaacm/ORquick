import numpy as np

A_ = [[7, 5, 9, 8, 11],
      [9, 12, 7, 11, 9],
      [8, 5, 4, 6, 9],
      [7, 3, 6, 9, 6],
      [4, 6, 7, 5, 11]]

# A_ = [[4, 8, 7, 15, 12],
#       [7, 9, 17, 14, 10],
#       [6, 9, 12, 8, 7],
#       [6, 7, 14, 6, 10],
#       [6, 9, 12, 10, 6]]

A = np.array(A_, np.float)

min_row = A.min(axis=1)
A = A - min_row.reshape(len(min_row), 1)
A = A - A.min(axis=0)
print(A)
while True:
    zeros = np.argwhere(A == 0)
    d_zeros = zeros[0].reshape(1, 2)
    nd_zeros = np.array([], dtype=np.int).reshape(0, 2)
    for zero in zeros[1:]:
        if zero[0] not in d_zeros[:, 0] and zero[1] not in d_zeros[:, 1]:
            d_zeros = np.concatenate((d_zeros, zero.reshape(1, 2)), axis=0)
        else:
            nd_zeros = np.concatenate((nd_zeros, zero.reshape(1, 2)), axis=0)
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
    print(A)


print(d_zeros)