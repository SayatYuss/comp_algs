def lagrange_interpolation(x_points, y_points, x):
    n = len(x_points)
    result = 0.0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if j != i:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Пример данных
x_points = [1, 2, 3]
y_points = [2, 3, 12]

# Точка, в которой нужно найти значение функции
x = 2.5

# Вычисление интерполированного значения
value = lagrange_interpolation(x_points, y_points, x)
print(f"f({x}) ≈ {value}")

