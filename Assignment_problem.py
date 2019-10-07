import numpy as np

A_ = [[7, 5, 9, 8, 11],
      [9, 12, 7, 11, 9],
      [8, 5, 4, 6, 9],
      [7, 3, 6, 9, 6],
      [4, 6, 7, 5, 11]]

A = np.array(A_, np.float)

min_row = A.min(axis=1)
A = A - min_row.reshape(len(min_row), 1)
A = A - A.min(axis=0)
print(A)
zeros = np.argwhere(A == 0)
# dep_zeros = []
# for i in range(A.shape[0]):
#     if np.sum(A[i] == 0) == 1:
#         dep_zeros.append([i, A[i].argmin()])
zeros_list = zeros.tolist()
dep_zeros = []
non_dep_zeros = []
i = 0
# for i in range(len(zeros[:, 0])):
while i < len(zeros[:, 0]):
    j = i
    if zeros[:, 0].tolist().count(zeros[i][0]) == 1:
        # dep_zeros.append(zeros_list[i])
        # for j in range(i, len(zeros[:, 0])):

        while j < len(zeros[:, 0]):
            print(len(zeros[:, 0]))
            if zeros[i, 1] == zeros[j, 1] and i != j:
                non_dep_zeros.append(zeros[j])
                no_del = zeros[:, 1] != zeros[j, 1]
                no_del[j] == True
                print(no_del)
                
                zeros = zeros[no_del]
            j += 1
    i += 1
