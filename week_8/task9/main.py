import numpy as np

epsilon = 0.1e-10   # Заданная погрешность
k_max = 10000     # Максимальное число итераций

# --- Функции ---

def create_matrix_variant_10(n, p, q, b):
    A = np.zeros((n, n))
    
    for i in range(n):  # i от 0 до n-1
        for j in range(n):  # j от 0 до n-1
            
            # Переходим к 1-индексации для расчетов
            i_calc = i + 1
            j_calc = j + 1
            
            if i == j:
                A[i, i] = 13 * (i_calc**(p / 3))
            else:
                numerator = abs(i_calc**p - j_calc**p)
                A[i, j] = b * (numerator**(-1 / q))
                
    return A


def power_method(A, epsilon, k_max):
    n = A.shape[0]
    
    # 1. Задаем начальный случайный вектор x(0)
    x_prev = np.random.rand(n) 
    
    lambda_curr = 0.0
    k = 0
    
    print(f"Начало итераций (max {k_max} итераций, epsilon {epsilon}):")

    # 2. Запускаем итерационный цикл
    while k < k_max:
        k += 1
        lambda_prev = lambda_curr
        
        
        # ||x^(k-1)|| - Евклидова норма (длина) вектора
        norm_x = np.linalg.norm(x_prev) 
        
        # e^(k-1) = x^(k-1) / ||x^(k-1)|| - Нормализация
        if norm_x == 0:
            print("Ошибка: норма вектора равна 0. Прерывание.")
            return 0, k # Возвращаем 0, чтобы избежать деления на ноль
            
        e = x_prev / norm_x 
        
        # x^k = A * e^(k-1)
        x_curr = np.dot(A, e) 
        
        # λ^k = (x^k, e^(k-1)) - Скалярное произведение
        lambda_curr = np.dot(x_curr, e) 
        
        # 3. Проверяем условие остановки
        if abs(lambda_curr - lambda_prev) < epsilon:
            print(f"Точность {epsilon} достигнута.")
            break
            
        x_prev = x_curr
    
    if k == k_max:
        print(f"За {k_max} итераций точность не достигнута.")
        
    return lambda_curr, k

# --- Выполнение ---

print("--- ЗАПУСК РАСЧЕТОВ ДЛЯ ВСЕХ КОМБИНАЦИЙ n, p, q ---")

n_range = range(5, 11) # от 5 до 10 вкл. (5, 6, 7, 8, 9, 10)
p_range = range(1, 5)  # от 1 до 4 вкл. (1, 2, 3, 4)
q_range = range(1, 5)  # от 1 до 4 вкл. (1, 2, 3, 4)

b_min = 0.01
b_max = 0.1

for n in n_range:
    for p in p_range:
        for q in q_range:
            
            b = np.random.uniform(b_min, b_max)
            
            print(f"\n=======================================================")
            print(f"--- НОВЫЙ ЗАПУСК: n={n}, p={p}, q={q}, b={b:.4f} ---")
            print(f"=======================================================")
            
            # 1. Генерируем матрицу
            A_variant = create_matrix_variant_10(n, p, q, b)

            # 2. Выполняем метод
            lambda_result, k_result = power_method(A_variant, epsilon, k_max)

            # 3. Выводим отчет для этой комбинации
            print(f"\n--- Результаты для n={n}, p={p}, q={q}, b={b:.4f} ---")
            # print("Исходная матрица A:")
            # print(A_variant)
            print(f"\nМаксимальное собственное значение λ_1: {lambda_result:.6f}")
            print(f"Число итераций k: {k_result}")
            print("=======================================================\n")