"""
Finding the largest root of the nonlinear equation 3x−cosx−1=0 using interpolation
(utilizing the Lagrange and Newton interpolation polynomials constructed with 10 equally spaced nodes)
With the nodes from the first part of a project, build a natural cubic interpolation spline.
Print the plot of the function you are interpolating, the interpolation polynomials (from the first part) and the cubic spline
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, simplify, solve

x = np.linspace(-25, 25, 1000)
y = 3 * x - np.cos(x) - 1

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='y = 3x - cos(x) - 1')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графік рівняння y= 3x - cos(x) - 1')
plt.grid(True)
plt.show()

nodes = np.linspace(-2, 2, 10)
print(nodes)
def f(x):
    return 3 * x - np.cos(x) - 1
values = f(nodes)
y = symbols('y')
# інтерполяційний поліном Лагранжа
def lagrange_interpolation(nodes, values):
    result = 0
    for i in range(len(nodes)):
        term = nodes[i]
        for j in range(len(nodes)):
            if j != i:
                term = term * (y - values[j]) / (values[i] - values[j])
        result += term
    return result
# інтерполяційний поліном Ньютона
def divided_differences(nodes, values):
    n = len(nodes)
    table = [[0] * n for _ in range(n)]

    for i in range(n):
        table[i][0] = nodes[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (values[i + j] - values[i])

    return table

def newton_interpolation(nodes, values):
    x = symbols('x')
    n = len(nodes)
    polynomial = nodes[0]
    differences_table = divided_differences(nodes, values)

    for i in range(1, n):
        term = 1
        for j in range(i):
            term *= (y - values[j])
        polynomial += differences_table[0][i] * term

    return simplify(polynomial)

newton_poly = newton_interpolation(nodes, values)
newton_poly_simplified = simplify(newton_poly)
lagrange_poly = lagrange_interpolation(nodes, values)
lagrange_poly_simplified = simplify(lagrange_poly)
print("Обернений поліном Ньютона:", newton_poly_simplified)
print("Обернений поліном Лагранжа:", lagrange_poly_simplified)
print("Корінь нелінійного рівняння із застосуванням оберненої інтерполяції (поліном Ньютона):", newton_poly_simplified.subs(y, 0))
print("Корінь нелінійного рівняння із застосуванням оберненої інтерполяції (поліном Лагранжа):", lagrange_poly_simplified.subs(y, 0))

#######################################################################

def original_function(x):
    return 3 * x - np.cos(x) - 1
def newton_polinom(x):
    return 5.38984004996856e-17*x**9 - 2.2647693446571e-5*x**8 - 4.3820992279158e-16*x**7 + 0.00138330152572559*x**6 + 1.07540497737515e-15*x**5 - 0.041661262627466*x**4 - 7.41627902643711e-16*x**3 + 0.499998370629544*x**2 + 3.0*x - 1.99999993205569
def lagrange_polinom(x):
    return -5.55111512312578e-17*x**9 - 2.26476934475361e-5*x**8 + 1.4432899320127e-15*x**7 + 0.00138330152571781*x**6 - 3.99680288865056e-14*x**5 - 0.0416612626274433*x**4 - 1.23234755733392e-14*x**3 + 0.499998370629551*x**2 + 3.0*x - 1.99999993205569
nodes = np.linspace(-2, 2, 10)
values = original_function(nodes)

h = np.diff(nodes)
delta_y = np.diff(values)

# другі похідні
alpha = np.zeros(len(nodes))
alpha[1:-1] = 3/h[1:] * delta_y[1:] - 3/h[:-1] * delta_y[:-1]

l, u, z = np.ones(len(nodes)), np.ones(len(nodes)), np.ones(len(nodes))
l[1] = 0
u[-1] = 0

for i in range(1, len(nodes)-1):
    l[i] = 2 * (nodes[i+1] - nodes[i-1]) - h[i-1] * z[i-1]
    z[i] = h[i] / l[i]
    u[i] = (alpha[i] - h[i-1] * u[i-1]) / l[i]

s_prime2 = np.zeros(len(nodes))
for i in range(len(nodes)-2, -1, -1):
    s_prime2[i] = z[i] * s_prime2[i+1] + u[i]

# коефіцієнти
a = values[:-1]
b = (delta_y - h * (2 * s_prime2[:-1] + s_prime2[1:])) / h
c = s_prime2[:-1]
d = (s_prime2[1:] - s_prime2[:-1]) / (3 * h)

x_values = np.linspace(-2, 2, 100)

# сплайн
y_spline = []
for x in x_values:
    segment_idx = np.searchsorted(nodes, x) - 1
    segment_x = x - nodes[segment_idx]
    y_spline.append(a[segment_idx] + b[segment_idx] * segment_x + c[segment_idx] * segment_x**2 + d[segment_idx] * segment_x**3)

# графіки
plt.plot(x_values, original_function(x_values), label='Original Function')
plt.plot(x_values, newton_polinom(x_values), label='Newton polinom')
plt.plot(x_values, lagrange_polinom(x_values), label='Lagrange polinom')

plt.plot(x_values, y_spline, label='Cubic Spline Interpolation', linestyle='dashed')

plt.scatter(nodes, values, color='red', label='Nodes')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# сплайн для кожного сегменту
for i in range(len(a)):
    print(f"Segment {i+1}: y = {a[i]:.2f} + {b[i]:.2f} * (x - {abs(nodes[i]):.2f}) + {c[i]:.2f} * (x - {abs(nodes[i]):.2f})^2 + {d[i]:.2f} * (x - {abs(nodes[i]):.2f})^3")

