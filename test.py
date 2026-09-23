import numpy as np

M = 10
K = 100
N = 10

A = np.zeros((M, K))
B = np.zeros((K, N))

def init_matrix(matrix):
    matrix[::2, :] = 0.005
    matrix[1::2, :] = 0.004

init_matrix(A)
init_matrix(B)

print(f"A @ B = \n {A @ B}")