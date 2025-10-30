import math
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np
import time


def main():
    np.random.seed(42)

    # --- генерим случайные, но воспроизводимые значения ---
    a = np.random.uniform(0.001, 1.0)
    b = np.random.uniform(0.001, 1.0)
    m = np.random.choice([2, 4, 6])
    n = np.random.choice([2, 4, 6])
    beta = np.random.uniform(0, 1)

    print(f"Используем параметры: a={a:.4f}, b={b:.4f}, m={m}, n={n}, beta={beta:.4f}")

    eps = 1e-4
    kmax = 10000
    d = math.sqrt(a**2 + b**2)
    alpha = math.sqrt(1 - beta**2)
    x = d
    y = d

    xs = [x]
    ys = [y]
    qs = []
    start_time = time.time()
    for k in range(1, kmax + 1):
        x_prev, y_prev = x, y
        
        x = minimize_scalar(
            lambda x_f: fxy(alpha, beta, a, b, x_f, y_prev, m, n),
            bounds=(-d,d)
        ).x

        y = minimize_scalar(
            lambda y_f: fxy(alpha, beta, a, b, x, y_f, m, n),
            bounds=(-d, d)
        ).x

        delta = math.sqrt((x - x_prev)**2 + (y - y_prev)**2)

        if delta < eps:
            print(f"Достигнута точность eps={eps} на итерации k={k}")
            break

        xs.append(x)
        ys.append(y)
        q_k = -math.log(delta) / math.log(10)
        qs.append(q_k)

        if k == kmax:
            print("Достигнут kmax")
            break
    end_time = time.time()

    elapsed_time =  end_time - start_time
    
    print(f"Всего итераций k={k}, выполнено за {elapsed_time} сек")
    print(f"x_min={x:.8f}")
    print(f"y_min={y:.8f}")
    print(f"Значение функции f(x,y): {fxy(alpha, beta, a, b, x, y, m, n):.8f}")

    plot_results(a, b, m, n, alpha, beta, d, xs, ys, qs, k, fxy)
    

def plot_results(a, b, m, n, alpha, beta, d, x_history, y_history, q_history, k_final, fxy_func):
    """
    Строит 4 графика по результатам минимизации.
    """
    print("\nПостроение графиков...")

    # --- 1. Подготовка сетки для 3D/контурного графика ---
    
    # Создаем диапазон значений x и y вокруг начальной точки
    plot_margin = 1.1
    x_grid = np.linspace(-d * plot_margin, d * plot_margin, 100)
    y_grid = np.linspace(-d * plot_margin, d * plot_margin, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Вычисляем Z (значение функции) для каждой точки сетки
    Z = fxy_func(alpha, beta, a, b, X, Y, m, n)

    # --- 2. Создание окна с 4 графиками ---
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Метод координатного спуска (a={a:.4f}, b={b:.4f}, β={beta:.4f}, m={m}, n={n})", fontsize=16)

    # [cite_start]--- График 1: 3D-поверхность функции f(x,y) [cite: 31] ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_title('График функции f(x,y)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')

    # --- График 2: Контурный график и траектория спуска (как Рис. 11.1) ---
    ax2 = fig.add_subplot(2, 2, 2)
    # Используем логарифмическую шкалу для уровней, чтобы лучше видеть "дно"
    levels = np.logspace(np.log10(Z.min() + 0.01), np.log10(Z.max()), 30)
    ax2.contour(X, Y, Z, levels=levels, cmap='viridis')
    # Рисуем траекторию спуска
    ax2.plot(x_history, y_history, 'ro-', markersize=3, linewidth=1, label='Траектория')
    ax2.set_title('Траектория спуска (Вид сверху)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # [cite_start]--- График 3: Сходимость x(k) и y(k) [cite: 32] ---
    ax3 = fig.add_subplot(2, 2, 3)
    # Ось k от 0 (начальная точка) до k_final
    k_axis_xy = range(k_final) 
    ax3.plot(k_axis_xy, x_history, 'b.-', label=f'x(k) -> {x_history[-1]:.6f}')
    ax3.plot(k_axis_xy, y_history, 'g.-', label=f'y(k) -> {y_history[-1]:.6f}')
    ax3.set_title('Сходимость x(k) и y(k)')
    ax3.set_xlabel('Итерация, k')
    ax3.set_ylabel('Значение')
    ax3.legend()
    ax3.grid(True)

    # [cite_start]--- График 4: Зависимость q(k) от k [cite: 32] ---
    ax4 = fig.add_subplot(2, 2, 4)
    # Ось k от 1 до k_final
    k_axis_q = range(1, k_final) 
    ax4.plot(k_axis_q, q_history, 'm.-')
    ax4.set_title('Зависимость q(k) от k')
    ax4.set_xlabel('Итерация, k')
    ax4.set_ylabel('q(k) = -lg(δ)')
    ax4.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("res.png")


def fxy(alpha, beta, a, b, x, y, m, n):
    slag1 = ((alpha * x - beta * y) / a) ** m
    slag2 = ((alpha * y + beta * x) / b) ** n

    return slag1 + slag2

if __name__ == '__main__':
    main()