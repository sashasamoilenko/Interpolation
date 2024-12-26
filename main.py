import matplotlib.pyplot as plt
import numpy as np
import math

x = np.linspace(-5, 15, 1000)
y = x**2 -np.cos(x)-1

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='y = x^3 - 3x^2 - 17x + 22')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графік рівняння y = x^3 - 3x^2 - 17x + 22')
plt.grid(True)
plt.show()
"""
print("Метод дихотомії")
#Метод дихотомії
def method_dyhotomii(f, a, b, E):
    n_aposteriorna = 0
    i=0
#Ітерації
    while (b - a) / 2 > E:
        c = (a + b) / 2
        print(f"x{i}= {c}")
        i += 1
        if f(c) == 0:
            return c, n_aposteriorna

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

        n_aposteriorna += 1

    root = (a + b) / 2
    return root, n_aposteriorna
#Функція
def f(x):
    return x**3 - 3*x**2 - 17*x + 22
#Проміжок та точність
a = 0
b = 3
E = 1e-4

root, n_aposteriorna = method_dyhotomii(f, a, b, E)
print(f"Корінь рівняння: {root}")
print(f"Кількість ітерацій за теоретичною апріорною оцінкою: {int(math.modf(math.log((b-a)/abs(E), 2))[1])}")
print(f"Кількість ітерацій за апостеріорною оцінкою: {n_aposteriorna}")
###########################################################################
print("Метод релаксації")
def method_relaxatsii(eq, E, tau):
    n_aposteriorna = 0
    x=x0
#Ітерації
    for i in range(1,100):
        x_new = x+0.08*eq(x)
        if abs(x_new - x) < E:
            return x_new, n_aposteriorna

        print(f"x{i}= {x_new}")
        x = x_new
        n_aposteriorna += 1
    return x, n_aposteriorna
#Функція
def eq(x):
    return x**3 - 3*x**2 - 17*x + 22

x0=0 #будь-яке з проміжку [0,3]
M1=20
m1=8
E = 1e-4
tau=2/(M1+m1)
q=(M1-m1)/(M1+m1)
root, n_aposteriorna = method_relaxatsii(eq, E, tau)
print(f"Корінь рівняння: {root}")
print(f"Кількість ітерацій за апріорною оцінкою: {int(math.modf((math.log(abs(x0-root)/E))/(math.log(1/q)))[1])+1}")
print(f"Кількість ітерацій за апостеріорною оцінкою: {n_aposteriorna}")
############################################################################
print("Метод простої ітерації")
def simple_iteration_method(phi, x0, E):
    x = x0
    n_aposteriorna = 0
    for i in range(1,100):
        x_new = phi(x)
        n_aposteriorna += 1

        if abs(x_new - x) < E:    #q=9/17 отже воно >1/2
            return x_new, n_aposteriorna

        print(f"x{i}= {x_new}")
        x = x_new

    return x, n_aposteriorna

def phi(x):
    return (x**3)/17-(3*x**2)/17+22/17

x0 = 0
E = 1e-4
q=9/17
root, n_aposteriorna = simple_iteration_method(phi, x0, E)

print(f"Корінь рівняння: {root}")
print(f"Кількість ітерацій за апріорною оцінкою: {int((math.modf((math.log((abs(phi(x0)-x0))/(E*(1-q))))/(math.log(1/q)))[1]))+1}")
print(f"Кількість ітерацій за апостеріорною оцінкою: {n_aposteriorna}")
"""