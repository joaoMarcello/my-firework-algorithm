from datetime import timedelta
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
from tqdm import trange

from fwa import FWA

class DiscreteFWA(FWA):
    def explode(self, fw, s_i, A_i):
        results = []
        for _ in range(s_i):
            spark = np.copy(fw)
            z = np.random.randint(1, self.dim + 1)
            idx = np.random.choice(self.dim, z, replace=False)
            for k in idx:
                delta = int(round(A_i * np.random.uniform(-1, 1)))
                spark[k] += delta
                spark[k] = int(round(self.clip(spark[k], self.bounds[k])))
            results.append(spark)
        return results

    def gaussian_explode(self):
        results = []
        for _ in range(self.m_hat):
            fw = random.choice(self.fireworks)
            spark = np.copy(fw)
            z = np.random.randint(1, self.dim + 1)
            idx = np.random.choice(self.dim, z, replace=False)
            g = int(round(norm.rvs(loc=1, scale=1)))
            for k in idx:
                spark[k] = int(round(spark[k] * g))
                spark[k] = int(round(self.clip(spark[k], self.bounds[k])))
            results.append(spark)
        return results

    def clip(self, val, bound):
        min_b, max_b = bound
        val = int(round(val))
        if val < min_b:
            return min_b
        elif val > max_b:
            return max_b
        return val

    def init_fireworks(self):
        self.fireworks = [
            np.array([np.random.randint(b[0], b[1] + 1) for b in self.bounds])
            for _ in range(self.n)
        ]
        self.best_solution = min(self.fireworks, key=self.func)
        self.best_value = self.func(self.best_solution)
        self.history.append(self.best_value)

    def init_fireworks_smart(self, smart_ratio=0.7):
        assert hasattr(self, "n_employees") and hasattr(self, "n_days"), \
            "Use set_problem_context() antes de usar init_fireworks_smart()."

        self.fireworks = []

        off_index = self.shift_id_to_index.get("OFF", 4)
        available_shift_ids = [i for i in range(len(self.shift_ids)) if i != off_index]

        n_smart = int(self.n * smart_ratio)
        n_random = self.n - n_smart

        for _ in range(n_smart):
            schedule = np.zeros((self.n_employees, self.n_days), dtype=float)

            for emp in range(self.n_employees):
                # Atribui todos os dias com turnos aleatórios (sem OFF)
                schedule[emp] = np.random.choice(available_shift_ids, size=self.n_days)

                # Define 1 folga em sábado ou domingo
                weekend_days = [d for d in range(self.n_days)
                                if (self.start_date + timedelta(days=d)).weekday() in [5, 6]]
                if weekend_days:
                    off_day = random.choice(weekend_days)
                    schedule[emp, off_day] = off_index

                # Define +2 folgas aleatórias em dias que ainda não são OFF
                working_days = [d for d in range(self.n_days)
                                if schedule[emp, d] != off_index]
                extra_offs = random.sample(working_days, k=min(2, len(working_days)))
                for d in extra_offs:
                    schedule[emp, d] = off_index

            self.fireworks.append(schedule.flatten())

        # Gera o restante da população de forma totalmente aleatória (mas discreta)
        for _ in range(n_random):
            schedule = np.zeros((self.n_employees, self.n_days), dtype=float)

            for emp in range(self.n_employees):
                # Turnos aleatórios entre os shift_ids válidos, incluindo OFF
                schedule[emp] = np.random.choice(range(len(self.shift_ids)), size=self.n_days)

            self.fireworks.append(schedule.flatten())


        self.best_solution = min(self.fireworks, key=self.func)
        self.best_value = self.func(self.best_solution)
        self.history.append(self.best_value)

