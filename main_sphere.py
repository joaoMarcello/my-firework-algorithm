import numpy as np
from fwa import FWA

def sphere(x):
    return np.sum(x ** 2)

dim = 10
bounds = [(-100, 100)] * dim

fwa = FWA(func=sphere, dim=dim, bounds=bounds)
fwa.config(n=5, m=50, max_iter=100)
fwa.run()

print("Melhor solução:", fwa.best_solution)
print("Valor:", fwa.func(fwa.best_solution))
