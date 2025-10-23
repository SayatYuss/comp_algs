import math
import numpy as np

a = 0.1
b = 1.0

def E0(xi):
    val = math.sqrt(xi) * (xi + math.sqrt(xi * (1 + xi)))
    val = max(-1, min(1, val))
    return math.asin(val)

# прямоугольники
def rectangle_method(f, a, b, m):
    h = (b - a) / m
    F = np.zeros(m)  
    for i in range(m):
        x = a + h * (i + 0.5)
        F[i] = f(x)
    I = h * np.sum(F)
    return I

# гаус
def gauss_method(f, a, b, n):
    t, A = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * t + 0.5 * (b + a)
    fx = np.array([f(xi) for xi in x])
    I = 0.5 * (b - a) * np.sum(A * fx)
    return I

print("Метод прямоугольников:")
for m in [5, 10, 20, 50, 100]:
    I_rect = rectangle_method(E0, a, b, m)
    print(f"m = {m:3d} → I ≈ {I_rect:.6f}")

print("\nМетод Гаусса:")
for cm in range(5, 12):
    I_gauss = gauss_method(E0, a, b, cm)
    print(f"c_m = {cm:2d} → I ≈ {I_gauss:.6f}")
