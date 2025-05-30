#!/usr/bin/env python
# Created by "Thieu" at 11:23, 30/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np

class CLPSO:
    def __init__(self, obj_func, dim, pop_size=30, max_iter=1000, bounds=(-100, 100), refreshing_gap=7):
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb, self.ub = bounds
        self.refreshing_gap = refreshing_gap

        # Initialize particles
        self.X = np.random.uniform(self.lb, self.ub, (pop_size, dim))
        self.V = np.random.uniform(-abs(self.ub - self.lb), abs(self.ub - self.lb), (pop_size, dim))
        self.pbest = self.X.copy()
        self.pbest_fitness = self.evaluate(self.pbest)
        self.fitness = self.pbest_fitness.copy()
        self.stagnation = np.zeros(pop_size)  # Counter for refreshing gap

        # Learning probabilities Pc for each particle (from 0.05 to 0.5)
        self.Pc = 0.05 + 0.45 * np.linspace(0, 1, pop_size)

        # Exemplar: dimension-wise best particle IDs for each particle
        self.exemplar = np.random.randint(0, pop_size, (pop_size, dim))
        self.update_exemplar()

    def evaluate(self, X):
        return np.apply_along_axis(self.obj_func, 1, X)

    def update_exemplar(self):
        for i in range(self.pop_size):
            for d in range(self.dim):
                if np.random.rand() < self.Pc[i]:
                    # Randomly select two particles (excluding itself)
                    idx = np.random.choice([x for x in range(self.pop_size) if x != i], 2, replace=False)
                    best_idx = idx[0] if self.pbest_fitness[idx[0]] < self.pbest_fitness[idx[1]] else idx[1]
                    self.exemplar[i, d] = best_idx
                else:
                    self.exemplar[i, d] = i

        # If all dimensions are from itself, enforce at least one dimension from another
        for i in range(self.pop_size):
            if np.all(self.exemplar[i] == i):
                d = np.random.randint(0, self.dim)
                self.exemplar[i, d] = np.random.choice([x for x in range(self.pop_size) if x != i])

    def optimize(self):
        for t in range(self.max_iter):
            # Velocity and position update
            r = np.random.rand(self.pop_size, self.dim)
            exemplar_best = self.pbest[self.exemplar, np.arange(self.dim)]
            self.V = 0.729 * self.V + 1.49445 * r * (exemplar_best - self.X)
            self.X += self.V

            # Enforce bounds
            self.X = np.clip(self.X, self.lb, self.ub)

            # Fitness evaluation
            new_fitness = self.evaluate(self.X)
            improved = new_fitness < self.pbest_fitness
            self.pbest[improved] = self.X[improved]
            self.pbest_fitness[improved] = new_fitness[improved]

            # Refreshing logic
            self.stagnation[improved] = 0
            self.stagnation[~improved] += 1
            for i in range(self.pop_size):
                if self.stagnation[i] >= self.refreshing_gap:
                    self.update_exemplar()
                    self.stagnation[i] = 0

            # Report progress
            if (t + 1) % 100 == 0 or t == 0:
                print(f"Iteration {t+1}/{self.max_iter} | Best Fitness: {self.pbest_fitness.min()}")

        best_idx = np.argmin(self.pbest_fitness)
        return self.pbest[best_idx], self.pbest_fitness[best_idx]

# Example: Rastrigin Function (multimodal test function)
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Run CLPSO
if __name__ == "__main__":
    dim = 30
    clpso = CLPSO(obj_func=rastrigin, dim=dim, pop_size=40, max_iter=1000, bounds=(-5.12, 5.12))
    best_sol, best_fit = clpso.optimize()
    print("\nBest Solution:", best_sol)
    print("Best Fitness:", best_fit)
