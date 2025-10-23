import numpy as np
import matplotlib.pyplot as plt

# --- 1. Функция по твоему варианту ---
def y_func(x, a, b, k, m):
    return (a + b * x**(1/m))**(-k)

# --- 2. Таблица конечных разностей ---
def forward_differences(y_nodes):
    diffs = [y_nodes]
    while len(diffs[-1]) > 1:
        diffs.append(np.diff(diffs[-1]))
    return diffs

# --- 3. Интерполяция Ньютона 2-го порядка в полуцелых точках ---
def newton_second_order(x_nodes, y_nodes):
    h = x_nodes[1] - x_nodes[0]
    diffs = forward_differences(y_nodes)
    delta1 = diffs[1]
    delta2 = diffs[2]

    xs_mid = []
    P_vals = []
    for i in range(len(y_nodes) - 2):  # используем i, i+1, i+2
        x_mid = x_nodes[i] + h/2
        s = (x_mid - x_nodes[i]) / h  # = 0.5
        P = y_nodes[i] + s*delta1[i] + (s*(s-1)/2)*delta2[i]
        xs_mid.append(x_mid)
        P_vals.append(P)
    return np.array(xs_mid), np.array(P_vals)

# --- 4. Подсчёт ошибок ---
def interpolation_errors(a, b, k, m, n):
    # строим узлы
    x_nodes = np.linspace(0, 1, n+1)
    y_nodes = y_func(x_nodes, a, b, k, m)

    # интерполяция
    xs_mid, P_vals = newton_second_order(x_nodes, y_nodes)
    y_true = y_func(xs_mid, a, b, k, m)

    # ошибки
    errors = np.abs(P_vals - y_true)
    emax = np.max(errors)
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)

    return emax, mse, rmse, xs_mid, P_vals, y_true, errors

# --- 5. Основная программа ---
if __name__ == "__main__":
    # параметры варианта (можешь менять)
    a, b, k, m = 2, 4, 1, 5
    n_values = [20, 40, 60, 80, 100]

    results = []
    for n in n_values:
        emax, mse, rmse, xs_mid, P_vals, y_true, errors = interpolation_errors(a, b, k, m, n)
        results.append((n, emax, mse, rmse))
        print(f"n={n}: emax={emax:.2e}, mse={mse:.2e}, rmse={rmse:.2e}")

    # график зависимости ошибок от n
    ns = [r[0] for r in results]
    emaxs = [r[1] for r in results]
    rmses = [r[3] for r in results]

    plt.figure(figsize=(8,5))
    plt.plot(ns, emaxs, 'o-', label="εmax")
    plt.plot(ns, rmses, 's-', label="RMSE")
    plt.xlabel("n (число интервалов)")
    plt.ylabel("Ошибка")
    plt.title("Зависимость ошибок интерполяции от n")
    plt.legend()
    plt.grid(True)
    plt.savefig("grafik.png", dpi=500, bbox_inches="tight")
