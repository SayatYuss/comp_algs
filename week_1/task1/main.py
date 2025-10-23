def taylor(x, epsilon=1e-10):
    if x <= 0:
        raise ValueError("ln(x) определён только для x > 0")

    y = (x - 1) / (x + 1)
    result = 0.0
    n = 0
    while True:
        term = (y ** (2 * n + 1)) / (2 * n + 1)
        result += term
        if abs(term) < epsilon:
            break
        n += 1
    return 2 * result

def log_a_b(a, b, epsilon=1e-10):
    return taylor(b, epsilon) / taylor(a, epsilon)

a = float(input("Введите основание логарифма a: "))
b = float(input("Введите число b: "))

print(f"log_{a}({b}) = {log_a_b(a, b):.4f}")