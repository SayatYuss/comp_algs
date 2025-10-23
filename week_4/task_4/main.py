import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Функция исходная
def f_original(x, sign, q):
    return x*(1-x) + sign * np.power(x, 1.0/q)

# кусочная аппроксимация
def piecewise_linear_interp(x_nodes, y_nodes, x_eval):
    return np.interp(x_eval, x_nodes, y_nodes)

# аппроксимация по полгонам
def poly_least_squares(x_nodes, y_nodes, deg, x_eval):
    deg_eff = min(deg, len(x_nodes)-1)  
    p = np.polyfit(x_nodes, y_nodes, deg_eff)
    return np.polyval(p, x_eval)

# параметры
qs = [2,3,4,5]
ns = [10,20,50]
x_dense = np.linspace(0,1,2001)

results = []

for q in qs:
    for sign_label, sign in [('+', 1), ('-', -1)]:
        for n in ns:
            # узлы
            x_nodes = np.linspace(0,1,n+1)
            y_nodes = f_original(x_nodes, sign, q)

            # аппроксимация
            y_true = f_original(x_dense, sign, q)
            y_pl = piecewise_linear_interp(x_nodes, y_nodes, x_dense)
            y_poly = poly_least_squares(x_nodes, y_nodes, deg=5, x_eval=x_dense)

            # ошибки
            err_pl = np.abs(y_true - y_pl)
            err_poly = np.abs(y_true - y_poly)

            max_err_pl = np.max(err_pl)
            max_err_poly = np.max(err_poly)

            l2_err_pl = np.sqrt(np.trapz(err_pl**2, x_dense))
            l2_err_poly = np.sqrt(np.trapz(err_poly**2, x_dense))

            results.append({
                'q': q,
                'sign': sign_label,
                'n': n,
                'max_err_piecewise': max_err_pl,
                'l2_err_piecewise': l2_err_pl,
                'max_err_poly_deg5': max_err_poly,
                'l2_err_poly_deg5': l2_err_poly,
            })

            # график
            plt.figure(figsize=(6,4))
            plt.plot(x_dense, y_true, label='Исходная функция')
            plt.plot(x_dense, y_pl, '--', label='Кусочно-линейная')
            plt.plot(x_dense, y_poly, ':', label='Полином степени 5 (МНК)')
            plt.scatter(x_nodes, y_nodes, s=20, c='red', zorder=3)
            plt.title(f'q={q}, знак {sign_label}, n={n}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            filename = f"graph_q{q}_sign{sign_label}_n{n}.png"
            plt.savefig(filename, dpi=500)
            plt.close()


# таблицы ошибок
df = pd.DataFrame(results).sort_values(['q','sign','n']).reset_index(drop=True)
print(df.to_string(index=False))