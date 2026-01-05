#!/usr/bin/env python
# Created by "Furkan Buyukyozgat" at 15:25, 05/01/2026-------%
#       Email: furkanbuyuky@gmail.com                        %
#       Github: https://github.com/furkanbuyuky              %
# -----------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalOSA(Optimizer):
    """
    Original Owl Search Algorithm (OSA).

    Main steps:
    - Normalize intensity using best/worst fitness
    - Compute distance to best
    - Intensity correction: ic = I / (r^2 + eps) + rand()
    - Update position: x_new = x +/- beta * ic * |alpha * best - x|
    - Apply bounds and greedy selection
    """

    def __init__(
        self,
        epoch=10000,
        pop_size=100,
        beta_max=1.9,
        alpha_max=0.5,
        eps=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 100000])

        self.beta_max = self.validator.check_float("beta_max", beta_max, (0.0, 10.0))
        self.alpha_max = self.validator.check_float("alpha_max", alpha_max, (0.0, 1.0))
        self.eps = float(eps)

        self.set_parameters(["epoch", "pop_size", "beta_max", "alpha_max"])
        self.sort_flag = False

        self.alpha = None

    def _beta(self, epoch: int) -> float:
        iter_ = epoch + 1
        return self.beta_max - self.beta_max * (iter_ / float(self.epoch))

    def _get_worst_agent_current_pop(self):
        fits = np.array([ag.target.fitness for ag in self.pop], dtype=float)
        if self.problem.minmax == "min":
            idx = int(np.argmax(fits))
        else:
            idx = int(np.argmin(fits))
        return self.pop[idx]

    def evolve(self, epoch: int) -> None:
        if self.alpha is None:
            self.alpha = self.generator.random() * self.alpha_max

        pvm = self.generator.random()
        beta = self._beta(epoch)

        bestowl = self.g_best
        weakowl = self._get_worst_agent_current_pop()

        denom = (weakowl.target.fitness - bestowl.target.fitness)
        if np.abs(denom) < self.eps:
            denom = self.eps

        for i in range(self.pop_size):
            x = self.pop[i].solution

            intensity = (self.pop[i].target.fitness - bestowl.target.fitness) / denom
            r = np.linalg.norm(x - bestowl.solution)
            ic = (intensity / (r * r + self.eps)) + self.generator.random()

            step = beta * ic * np.abs(self.alpha * bestowl.solution - x)
            if pvm < 0.5:
                x_new = x + step
            else:
                x_new = x - step

            x_new = self.correct_solution(x_new)
            agent_new = self.generate_agent(x_new)

            if self.compare_target(agent_new.target, self.pop[i].target, self.problem.minmax):
                self.pop[i] = agent_new
