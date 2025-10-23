import numpy as np

# --- Данные для задачи ---
n = 10
np.random.seed(42)

A = np.random.randint(10, 31, size=(n, n)).astype(float)

x_true = np.random.randint(1, 10, size=n).astype(float)

b = A @ x_true

# --- Метод Гаусса ---
def gauss_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    if A.shape != (n, n):
        raise ValueError("Размеры матрицы A и вектора b не совпадают")
    
    Ab = np.column_stack((A, b))
    
    # Прямой ход
    for i in range(n):
        max_row = i
        for j in range(i + 1, n):
            if abs(Ab[j, i]) > abs(Ab[max_row, i]):
                max_row = j
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        
        if abs(Ab[i, i]) < 1e-10:
            raise ValueError("Матрица вырожденная или близка к вырожденной")
        
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    
    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
    
    return x

# --- Красивый вывод ---
def print_matrix(A, b):
    print("\nМатрица маршрутов A и потребности магазинов b:")
    for i in range(len(A)):
        row_str = " ".join(f"{val:5.1f}" for val in A[i])
        print(f"[ {row_str} ]  | {b[i]:5.1f}")

# --- Основной блок ---
if __name__ == "__main__":
    try:
        print_matrix(A, b)

        solution = gauss_elimination(A, b)

        print("\nСколько каждая машина доставляет (решение системы):")
        for i, val in enumerate(solution, 1):
            print(f"x{i:2d} = {val:8.2f}")

        residual = np.dot(A, solution) - b
        print("\nПроверка (A*x - b), остатки должны быть близки к 0:")
        for i, r in enumerate(residual, 1):
            print(f"r{i:2d} = {r:10.2e}")

    except ValueError as e:
        print(f"Ошибка: {e}")
