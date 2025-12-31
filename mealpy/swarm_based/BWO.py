import math
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalBWO(Optimizer):
    """
    Beluga Whale Optimization (BWO)
    Paper-faithful implementation adapted to Mealpy.
    """

    def __init__(self, epoch=1000, pop_size=50, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    # ===== Levy flight from your original code =====
    def _levy_flight(self, dim):
        beta = 1.5
        sigma = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
        ) ** (1 / beta)

        u = self.generator.normal(0.0, sigma, size=dim)
        v = self.generator.normal(0.0, 1.0, size=dim)
        step = u / (np.abs(v) ** (1.0 / beta))
        return 0.05 * step

    # ===== One iteration =====
    def evolve(self, epoch):
        lb, ub = self.problem.lb, self.problem.ub
        n = self.pop_size
        dim = self.problem.n_dims

        B0 = self.generator.random(n)
        Bf = B0 * (1.0 - epoch / (2.0 * self.epoch))

        pop_new = []

        for i in range(n):
            r = self.generator.integers(0, n)
            while r == i:
                r = self.generator.integers(0, n)

            Xi = self.pop[i].solution.copy()

            if Bf[i] > 0.5:
                r1, r2 = self.generator.random(), self.generator.random()
                for j in range(dim):
                    pj = self.generator.integers(0, dim)
                    p1 = self.generator.integers(0, dim)
                    base = Xi[pj]
                    diff = self.pop[r].solution[p1] - base
                    trig = math.sin(2 * math.pi * r2) if (j + 1) % 2 == 0 else math.cos(2 * math.pi * r2)
                    Xi[j] = base + diff * (1 + r1) * trig
            else:
                r3, r4 = self.generator.random(), self.generator.random()
                C1 = 2.0 * r4 * (1.0 - epoch / self.epoch)
                LF = self._levy_flight(dim)
                Xi = r3 * self.g_best.solution - r4 * Xi + C1 * LF * (
                    self.pop[r].solution - Xi
                )

            Xi = self.correct_solution(Xi)
            agent = self.generate_empty_agent(Xi)
            agent.target = self.get_target_wrapper(Xi)
            pop_new.append(agent)

        self.pop = self.greedy_selection_population(self.pop, pop_new)

        # ===== Whale fall (Eq. 8–10) =====
        Wf = 0.1 - 0.05 * (epoch / self.epoch)
        C2 = 2.0 * Wf * n
        X_step = (ub - lb) * math.exp(-C2 * epoch / self.epoch)

        for i in range(n):
            if self.generator.random() < Wf:
                r = self.generator.integers(0, n)
                while r == i:
                    r = self.generator.integers(0, n)
                r5, r6, r7 = self.generator.random(3)
                Xi = (
                    r5 * self.pop[i].solution
                    - r6 * self.pop[r].solution
                    + r7 * X_step
                )
                Xi = self.correct_solution(Xi)
                agent = self.generate_empty_agent(Xi)
                agent.target = self.get_target_wrapper(Xi)
                if agent.target.fitness < self.pop[i].target.fitness:
                    self.pop[i] = agent
