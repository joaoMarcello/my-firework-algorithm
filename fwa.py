import numpy as np
import random
from scipy.stats import norm
from tqdm import trange

class FWA:
    def __init__(self, func, dim, bounds):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.fireworks = []
        self.best_solution = None
        self.best_value = None
        self.history = []

    def load_prob(self, n=5, m=50, a=0.04, b=0.8, A_hat=40, m_hat=5, max_iter=100):
        self.n = n
        self.m = m
        self.a = a
        self.b = b
        self.A_hat = A_hat
        self.m_hat = m_hat
        self.max_iter = max_iter

    def init_fireworks(self):
        self.fireworks = [
            np.random.uniform(low=[b[0] for b in self.bounds],
                              high=[b[1] for b in self.bounds])
            for _ in range(self.n)
        ]
        self.best_solution = min(self.fireworks, key=self.func)
        self.best_value = self.func(self.best_solution)
        self.history.append(self.best_value)

    def run(self):
        self.init_fireworks()
        for _ in trange(self.max_iter, desc="FWA"):
            self.iter()

    def iter(self):
        sparks = []
        f_values = [self.func(fw) for fw in self.fireworks]
        ymax, ymin = max(f_values), min(f_values)
        total_diff = sum(ymax - f for f in f_values) + 1e-12

        for i, fw in enumerate(self.fireworks):
            s_i = self.m * (ymax - f_values[i] + 1e-12) / total_diff
            s_i = round(np.clip(s_i, self.a * self.m, self.b * self.m))
            A_i = self.A_hat * (f_values[i] - ymin + 1e-12) / (sum(f - ymin for f in f_values) + 1e-12)
            sparks += self.explode(fw, s_i, A_i)

        sparks += self.gaussian_explode()
        self.fireworks = self.select(sparks + self.fireworks)
        best = min(self.fireworks, key=self.func)

        best_val = self.func(best)

        if best_val < self.best_value:
            self.best_solution = best
            self.best_value = best_val

        self.history.append(self.best_value)

    def explode(self, fw, s_i, A_i):
        results = []
        for _ in range(s_i):
            spark = np.copy(fw)
            z = np.random.randint(1, self.dim + 1)
            idx = np.random.choice(self.dim, z, replace=False)
            for k in idx:
                h = A_i * np.random.uniform(-1, 1)
                spark[k] += h
                spark[k] = self.clip(spark[k], self.bounds[k])
            results.append(spark)
        return results

    def gaussian_explode(self):
        results = []
        for _ in range(self.m_hat):
            fw = random.choice(self.fireworks)
            spark = np.copy(fw)
            z = np.random.randint(1, self.dim + 1)
            idx = np.random.choice(self.dim, z, replace=False)
            g = norm.rvs(loc=1, scale=1)
            for k in idx:
                spark[k] *= g
                spark[k] = self.clip(spark[k], self.bounds[k])
            results.append(spark)
        return results

    # def clip(self, val, bound):
    #     min_b, max_b = bound
    #     if val < min_b or val > max_b:
    #         return min_b + abs(val) % (max_b - min_b)
    #     return val

    def clip(self, val, bound):
        # Implementação reflexiva baseada no paper original (reflete no limite)
        min_b, max_b = bound
        while val < min_b or val > max_b:
            if val < min_b:
                val = min_b + (min_b - val)
            elif val > max_b:
                val = max_b - (val - max_b)
        return val


    def select(self, candidates):
        f_vals = [self.func(x) for x in candidates]
        best = candidates[np.argmin(f_vals)]
        distances = np.array([sum(np.linalg.norm(x - y) for y in candidates) for x in candidates])
        probs = distances / distances.sum()

        # selected = [best] + list(np.random.choice(candidates, self.n - 1, p=probs, replace=False))

        indices = np.arange(len(candidates))
        chosen = np.random.choice(indices, self.n - 1, p=probs, replace=False)
        selected = [best] + [candidates[i] for i in chosen]

        return selected
