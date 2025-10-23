import numpy as np

a, b, c = 2, 1.5, 1


np.random.seed(42)

alphas = np.random.uniform(-5, 5, 3)
betas = np.random.uniform(0.5, 4, 3)
pis = np.random.uniform(0.1, 0.5, 3)
qs = np.random.uniform(0.1, 2, 3)

def integrand(x):
    abs_diff = np.abs(x - alphas)
    return np.prod((abs_diff ** betas) * np.exp(-pis * (abs_diff ** qs)))

def monte_carlo_integral(N):
    xs = np.random.uniform([-a, -b, -c], [a, b, c], size=(N, 3))
    inside = ((xs[:,0]/a)**2 + (xs[:,1]/b)**2 + (xs[:,2]/c)**2) <= 1
    vals = np.array([integrand(x) for x in xs[inside]])
    mean_f = vals.mean() if vals.size > 0 else 0
    V_ellipsoid = (4/3) * np.pi * a * b * c
    return mean_f * V_ellipsoid


for N in [1000, 5000, 10000, 100000, 1000000]:
    I = monte_carlo_integral(N)
    print(f"N={N}, интеграл ≈ {I:.4f}")
