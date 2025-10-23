import numpy as np
import matplotlib.pyplot as plt

def formula(x, m, j):
    return (1 - x) ** (1 / m) - 2 * (x ** (1 / j))

m = 2
j = 3
step = 10
x1 = 0
x2 = 1

x_values = np.linspace(x1, x2, step)
y_values = [formula(x, m, j) for x in x_values]

for x in np.linspace(x1, x2, step):
    try:
        print(f"При x = {x:.4f}, ответ = {formula(x, m, j):.5f}")
    except Exception as e:
        print(f"Ошибка при расчете x = {x}\nОшибка {e}")

# построение графика
plt.plot(x_values, y_values, label=f"f(x), m={m}, j={j}, шаг = {step}", color="blue")
plt.title("График функции f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

# сохранить график в файл
plt.savefig("grafik.png", dpi=500, bbox_inches="tight")

print("График сохранён в файл grafik.png")
