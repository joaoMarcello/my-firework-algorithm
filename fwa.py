from datetime import timedelta
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
from tqdm import trange


class FWA:
    def __init__(self, func, dim, bounds, selection_method='distance', seed=None):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.fireworks = []
        self.best_solution = None
        self.best_value = None
        self.history = []
        self.selection_method = selection_method
        self.n = None
        self.current_iter = 0

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def config(self, n=5, m=50, a=0.04, b=0.8, A_hat=40, m_hat=5, max_iter=100):
        self.n = n
        self.m = m
        self.a = a
        self.b = b
        self.A_hat = A_hat
        self.m_hat = m_hat
        self.max_iter = max_iter

    def set_problem_context(self, start_date, n_days, n_employees, shift_id_to_index, shift_ids):
        self.start_date = start_date
        self.n_days = n_days
        self.n_employees = n_employees
        self.shift_id_to_index = shift_id_to_index
        self.shift_ids = shift_ids

    
    def apply_joy_factor(self, J, J_hat):
        if J == 0 or J_hat == 0:
            return

        if not hasattr(self, "start_date"):
            raise ValueError("Você precisa configurar o contexto do problema com set_problem_context().")

        n_mutate = max(1, int(self.n * J))
        mutate_indices = np.random.choice(len(self.fireworks), size=n_mutate, replace=False)

        # Pré-computar os índices dos dias que são sábado (5) ou domingo (6)
        weekend_days = [
            day for day in range(self.n_days)
            if (self.start_date + timedelta(days=day)).weekday() in [5, 6]
        ]

        # Índice do turno de folga
        off_index = self.shift_id_to_index.get("OFF", 0)

        for idx in mutate_indices:
            fw = self.fireworks[idx]
            schedule = np.rint(fw).astype(int).reshape((self.n_employees, self.n_days))

            # Gera máscara booleana aleatória para finais de semana
            mutation_mask = np.random.rand(self.n_employees, len(weekend_days)) < J_hat

            # Aplica OFF nos finais de semana conforme a máscara
            for i, day in enumerate(weekend_days):
                schedule[:, day][mutation_mask[:, i]] = off_index

            # Atualiza firework original
            self.fireworks[idx] = schedule.flatten().astype(float)


    def init_fireworks(self):
        self.fireworks = [
            np.random.uniform(low=[b[0] for b in self.bounds],
                              high=[b[1] for b in self.bounds])
            for _ in range(self.n)
        ]
        self.best_solution = min(self.fireworks, key=self.func)
        self.best_value = self.func(self.best_solution)
        self.history.append(self.best_value)

    def run(self, verbose=False, log_freq=10):
        self.init_fireworks()
        self.current_iter = 0
        pbar = trange(self.max_iter, desc="FWA", dynamic_ncols=True)
        for i in pbar:
            self.iter()
            self.current_iter += 1
            pbar.set_description(f"FWA {self.best_value:.0f}")
            if verbose and (i % log_freq == 0 or i == self.max_iter - 1):
                print(f"Iter {i}: best = {self.best_value}")

    def get_dynamic_amplitude(self):
        initial = self.A_hat
        final = 0.2  # amplitude mínima no fim
        decay_rate = self.current_iter / (self.max_iter - 1 + 1e-8)
        return initial * (1 - decay_rate) + final * decay_rate


    def iter(self):
        sparks = []
        f_values = [self.func(fw) for fw in self.fireworks]
        ymax, ymin = max(f_values), min(f_values)
        total_diff = sum(ymax - f for f in f_values) + 1e-12

        amplitude_denom = sum(f - ymin for f in f_values) + 1e-12

        for i, fw in enumerate(self.fireworks):
            # cálculo da quant. de faíscas do fogo atual.
            # fogos com menor fitness geram mais faíscas
            s_i = self.m * (ymax - f_values[i] + 1e-12) / total_diff
            s_i = round(np.clip(s_i, self.a * self.m, self.b * self.m))

            # cálculo da amplitude da explosão. vai ser maior para fogos
            # com fitness ruins (vão explorar mais) e menor para fogos 
            # com fitness bons (busca será mais local)

            # A_i = self.A_hat * (f_values[i] - ymin + 1e-12) / amplitude_denom
            A_dynamic = self.get_dynamic_amplitude()
            A_i = A_dynamic * (f_values[i] - ymin + 1e-12) / amplitude_denom


            sparks += self.explode(fw, s_i, A_i)

        sparks += self.gaussian_explode()
        self.fireworks = self.select(sparks + self.fireworks)
        best = min(self.fireworks, key=self.func)

        best_val = self.func(best)

        if self.best_value is None or best_val < self.best_value:
            self.best_solution = best
            self.best_value = best_val

        self.history.append(self.best_value)

        self.apply_joy_factor(J=0.2, J_hat=0.5)


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
    #     # Implementação reflexiva baseada no paper original (reflete no limite)
    #     min_b, max_b = bound
    #     while val < min_b or val > max_b:
    #         if val < min_b:
    #             val = min_b + (min_b - val)
    #         elif val > max_b:
    #             val = max_b - (val - max_b)
    #     # Correção final para garantir limite exato em casos de imprecisão numérica
    #     return np.clip(val, min_b, max_b)
    
    def clip(self, val, bound):
        min_b, max_b = bound
        range_b = max_b - min_b
        if val < min_b or val > max_b:
            val = min_b + abs((val - min_b) % (2 * range_b))
            if val > max_b:
                val = max_b - (val - max_b)
        return val


    def select(self, candidates):
        candidates = np.array(candidates)

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
        best = candidates[best_idx]  # elitismo

        # Cálculo eficiente das distâncias entre candidatos
        distances = cdist(candidates, candidates).sum(axis=1)
        probs = distances / distances.sum()

        indices = np.arange(len(candidates))
        valid_indices = [i for i in indices if i != best_idx]

        if len(valid_indices) == 0:
            raise ValueError("Nenhum candidato disponível além do melhor.")

        probs_filtered = probs[valid_indices]
        probs_filtered /= probs_filtered.sum()

        selected = [best]  # sempre mantemos o melhor

        if len(valid_indices) < self.n - 1:
            # Poucos candidatos viáveis: permitimos repetição com ruído
            chosen = np.random.choice(valid_indices, self.n - 1, replace=True)
            for i in chosen:
                spark = np.copy(candidates[i])
                noise = np.random.normal(loc=0, scale=0.2, size=spark.shape)  # ruído suave
                spark += noise
                spark = np.clip(spark, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
                selected.append(spark)
        else:
            # Seleção por diversidade sem repetição
            chosen = np.random.choice(valid_indices, self.n - 1, p=probs_filtered, replace=False)
            selected += [candidates[i] for i in chosen]

        return selected

    
    def __select_roulette(self, candidates):
        f_vals = np.array([self.func(x) for x in candidates])
        best_idx = np.argmin(f_vals)
        best = candidates[best_idx]

        # Inverso da aptidão
        fitness_inv = np.max(f_vals) - f_vals + 1e-12

        # Zera a probabilidade do melhor (elitismo garantido fora da roleta)
        fitness_inv[best_idx] = 0.0

        # Renormaliza as probabilidades
        probs = fitness_inv / fitness_inv.sum()

        # Escolhe n - 1 sem repetição
        remaining_indices = np.random.choice(
            np.arange(len(candidates)),
            size=self.n - 1,
            replace=False,
            p=probs
        )

        selected = [best] + [candidates[i] for i in remaining_indices]
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
    
    def save_to_disc(self, path: str = "fwa_result.json"):
        # Salva a melhor solução, valor, histórico e parâmetros do problema em JSON.
        if self.best_solution is None:
            raise ValueError("Nenhuma solução foi encontrada ainda.")
        
        data = {
            "best_solution": self.best_solution.tolist(),
            "best_value": self.best_value,
            "history": self.history,
            "parameters": {
                "n": self.n,
                "m": self.m,
                "a": self.a,
                "b": self.b,
                "A_hat": self.A_hat,
                "m_hat": self.m_hat,
                "max_iter": self.max_iter,
                "dim": self.dim,
                # "bounds": [list(b) for b in self.bounds]
            }
        }

        # cria o diretório caso não exista
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        
        print(f"Resultados salvos em: {path}")


    def load_best(self, path: str = "fwa_result.json"):
        # Carrega melhor solução, valor, histórico e parâmetros de um arquivo JSON.
        with open(path, "r") as f:
            data = json.load(f)

        self.best_solution = np.array(data["best_solution"])
        self.best_value = data["best_value"]
        self.history = data["history"]

        params = data.get("parameters", {})
        self.n = params.get("n", None)
        self.m = params.get("m", None)
        self.a = params.get("a", None)
        self.b = params.get("b", None)
        self.A_hat = params.get("A_hat", None)
        self.m_hat = params.get("m_hat", None)
        self.max_iter = params.get("max_iter", None)
        self.dim = params.get("dim", None)
        # self.bounds = [tuple(b) for b in params.get("bounds", [])]

        print(f"Solução carregada de {path} com valor: {self.best_value}")


    def plot_history_from_file(self, json_path, save_path=None):

        self.load_best(path=json_path)

        if not self.history:
            print("Histórico vazio.")
            return

        plt.figure(figsize=(12, 5))
        plt.plot(self.history, label=f"Fitness (best: {self.best_value})")
        plt.title("Evolução da Função Objetivo (FWA)")
        plt.xlabel("Iteração")
        plt.ylabel("Valor da função objetivo")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        plt.show()

