import numpy as np

#Метод Якобі
def jacobi(A, b, x0, tol, max_iter):
    n = len(A)
    x = x0.copy()
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum_term = 0
            for j in range(n):
                if j != i:
                    sum_term += A[i, j] * x[j]
            x_new[i] = (b[i] - sum_term) / A[i, i]
        if k==1:
            print("Кубічна норма для першої ітерації", np.linalg.norm(x_new - x, ord=np.inf))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print("Кубічна норма для останньої ітерації", np.linalg.norm(x_new - x, ord=np.inf))
            print(f"Iteration {k}: {x_new}")
            return x_new
        x = x_new
        print(f"Iteration {k}: {x}")
    return x

A = np.array([[4, 1, -2, 1],
              [5, -6, 1, 0],
              [1, 3, 8, -2],
              [0, 2, -2, 5]], dtype=float)
b = np.array([4, -4, 23, 18], dtype=float)
x0 = np.zeros(4, dtype=float)
tolerance = 1e-6
max_iterations = 100
result = jacobi(A, b, x0, tolerance, max_iterations)
result = [round(x) for x in result]
print(result)


#Число обумовленості
import numpy as np
A = np.array([[4, 1, -2, 1],
              [5, -6, 1, 0],
              [1, 3, 8, -2],
              [0, 2, -2, 5]])
matrix_norm = np.linalg.norm(A, ord=np.inf)
A_inv = np.linalg.inv(A)
inverse_matrix_norm = np.linalg.norm(A_inv, ord=np.inf)
condition_number = matrix_norm * inverse_matrix_norm
print("Число обумовленості матриці A:", condition_number)


#Метод прогонки
def progonky(A, B, C, F):
    n = len(F)
    # коефіцієнти α та β
    alpha = [-C[0] / B[0]]
    beta = [F[0] / B[0]]

    print("Iteration 0: alpha[0] =", alpha[0], "beta[0] =", beta[0])
    # проміжні коефіцієнти α та β
    for i in range(1, n):
        alpha_i = -C[i] / (B[i] + A[i] * alpha[i - 1])
        beta_i = (F[i] - A[i] * beta[i - 1]) / (B[i] + A[i] * alpha[i - 1])

        alpha.append(alpha_i)
        beta.append(beta_i)

        print("Iteration", i, ": alpha[", i, "] =", alpha_i, "beta[", i, "] =", beta_i)

    # зворотний хід
    x = [0] * n
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x

A = [0, 1, 1, -3]  # Діагональна нижня матриця
B = [4, 2, 5, 3]  # Головна діагональ
C = [3, -1, -4, 0]  # Діагональна верхня матриця
F = [17, 6, 1, 3]  # Вектор вільних членів

result = progonky(A, B, C, F)
result = [round(x) for x in result]
print(result)


#Число обумовленості
import numpy as np
A = np.array([[4, 3, 0, 0],
              [1, 2, -1, 0],
              [0, 1, 5, -4],
              [0, 0, -3, 3]])
matrix_norm = np.linalg.norm(A, ord=np.inf)
A_inv = np.linalg.inv(A)
inverse_matrix_norm = np.linalg.norm(A_inv, ord=np.inf)
condition_number = matrix_norm * inverse_matrix_norm
print("Число обумовленості матриці A:", condition_number)

