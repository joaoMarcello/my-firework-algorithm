import numpy as np
import random
from scipy.stats import norm
from tqdm import trange

class FWA:
    def __init__(self, func, dim, bounds, selection_method='distance'):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.fireworks = []
        self.best_solution = None
        self.best_value = None
        self.history = []
        self.selection_method = selection_method
        self.n = None

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

    def run(self, verbose=False):
        self.init_fireworks()
        for i in trange(self.max_iter, desc="FWA"):
            self.iter()
            if verbose:
                print(f"Iter {i}: best = {self.best_value}")

    def iter(self):
        sparks = []
        f_values = [self.func(fw) for fw in self.fireworks]
        ymax, ymin = max(f_values), min(f_values)
        total_diff = sum(ymax - f for f in f_values) + 1e-12

        amplitude_denom = sum(f - ymin for f in f_values) + 1e-12

        for i, fw in enumerate(self.fireworks):
            s_i = self.m * (ymax - f_values[i] + 1e-12) / total_diff
            s_i = round(np.clip(s_i, self.a * self.m, self.b * self.m))
            A_i = self.A_hat * (f_values[i] - ymin + 1e-12) / amplitude_denom
            sparks += self.explode(fw, s_i, A_i)

        sparks += self.gaussian_explode()
        self.fireworks = self.select(sparks + self.fireworks)
        best = min(self.fireworks, key=self.func)

        best_val = self.func(best)

        if self.best_value is None or best_val < self.best_value:
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

    def clip(self, val, bound):
        # Implementação reflexiva baseada no paper original (reflete no limite)
        min_b, max_b = bound
        while val < min_b or val > max_b:
            if val < min_b:
                val = min_b + (min_b - val)
            elif val > max_b:
                val = max_b - (val - max_b)
        # return val
        return np.clip(val, min_b, max_b)

    def select(self, candidates):
        if self.selection_method == 'distance':
            return self.__select_distance(candidates)
        elif self.selection_method == 'roulette':
            return self.__select_roulette(candidates)
        elif self.selection_method == 'tournament':
            return self.__select_tournament(candidates)
        else:
            return self.__select_random(candidates)
        
    def __select_distance(self, candidates):
        f_vals = [self.func(x) for x in candidates]
        best_idx = np.argmin(f_vals)
        best = candidates[best_idx]

        distances = np.array([
            sum(np.linalg.norm(x - y) for y in candidates)
            for x in candidates
        ])
        probs = distances / distances.sum()
        indices = np.arange(len(candidates))

        # evitar duplicar o melhor
        chosen = [i for i in np.random.choice(indices, self.n - 1, p=probs, replace=False) if i != best_idx]
        while len(chosen) < self.n - 1:
            i = np.random.choice(indices, p=probs)
            if i != best_idx and i not in chosen:
                chosen.append(i)

        selected = [best] + [candidates[i] for i in chosen]
        return selected
    
    def __select_roulette(self, candidates):
        f_vals = [self.func(x) for x in candidates]
        best_idx = np.argmin(f_vals)
        best = candidates[best_idx]

        fitness_inv = np.max(f_vals) - np.array(f_vals) + 1e-12
        probs = fitness_inv / fitness_inv.sum()
        indices = np.arange(len(candidates))

        # evitar duplicar o melhor
        chosen = [i for i in np.random.choice(indices, self.n - 1, p=probs, replace=False) if i != best_idx]
        while len(chosen) < self.n - 1:
            i = np.random.choice(indices, p=probs)
            if i != best_idx and i not in chosen:
                chosen.append(i)

        selected = [best] + [candidates[i] for i in chosen]
        return selected
    
    def __select_tournament(self, candidates):
        f_vals = [self.func(x) for x in candidates]
        best_idx = np.argmin(f_vals)
        best = candidates[best_idx]

        selected = [best]
        candidates_indices = list(range(len(candidates)))
        while len(selected) < self.n:
            i1, i2 = random.sample(candidates_indices, 2)
            winner = candidates[i1] if f_vals[i1] < f_vals[i2] else candidates[i2]
            if winner not in selected:
                selected.append(winner)
        return selected
    
    def __select_random(self, candidates):
        f_vals = [self.func(x) for x in candidates]
        best_idx = np.argmin(f_vals)
        best = candidates[best_idx]

        indices = np.arange(len(candidates))

        # evitar duplicar o melhor
        chosen = [i for i in np.random.choice(indices, self.n - 1, replace=False) if i != best_idx]
        while len(chosen) < self.n - 1:
            i = np.random.choice(indices)
            if i != best_idx and i not in chosen:
                chosen.append(i)

        selected = [best] + [candidates[i] for i in chosen]
        return selected
