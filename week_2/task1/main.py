import numpy as np
import matplotlib.pyplot as plt

def formula(x, m, j):
    return (1 - x) ** (1 / m) - 2 * (x ** (1 / j))

def bisection(func, a, b, eps=1e-6, m=2, j=3):
    if func(a, m, j) * func(b, m, j) > 0:
        raise ValueError("На концах отрезка нет смены знака, метод не применим.")
    while (b - a) / 2 > eps:
        c = (a + b) / 2
        if func(c, m, j) == 0:
            return c
        if func(a, m, j) * func(c, m, j) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def derivative(func, x, m, j, h=1e-6):
    return (func(x + h, m, j) - func(x - h, m, j)) / (2 * h)

def newton(func, x0, eps=1e-6, m=2, j=3, max_iter=100):
    x = x0
    for _ in range(max_iter):
        f_x = func(x, m, j)
        df_x = derivative(func, x, m, j)
        if abs(df_x) < 1e-12:
            raise ValueError("Производная слишком мала, метод не работает.")
        x_new = x - f_x / df_x
        if abs(x_new - x) < eps:
            return x_new
        x = x_new
    return x

m, j = 1, 3
x1, x2 = 0, 1
step = 0.05

x_values = np.arange(x1, x2 + step, step)
y_values = [formula(x, m, j) for x in x_values]

for x in x_values:
    print(f"При x = {x:.2f}, f(x) = {formula(x, m, j):.3f}")

root_bisect = bisection(formula, x1, x2, m=m, j=j)
print(f"\nКорень методом дихотомии: {root_bisect.real:.6f}, f(x)={formula(root_bisect, m, j):.2e}")

root_newton = newton(formula, 0.5, m=m, j=j)
print(f"Корень методом Ньютона: {root_newton.real:.6f}, f(x)={formula(root_newton, m, j):.2e}")

plt.plot(x_values, y_values, label=f"f(x), m={m}, j={j}, шаг={step}", color="blue")
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(root_bisect, color="red", linestyle="--", label="Корень (дихотомия)")
plt.axvline(root_newton.real, color="green", linestyle="--", label="Корень (Ньютон)")

plt.title("График функции f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.savefig("grafik.png", dpi=500, bbox_inches="tight")

print("График сохранён в файл grafik.png")
