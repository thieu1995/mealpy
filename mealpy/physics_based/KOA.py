#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalKOA(Optimizer):
    """
    The original version of: Kepler Optimization Algorithm (KOA)

    Links:
        1. https://doi.org/10.1016/j.knosys.2023.110454

    Notes (parameters):
        1. T (int): cycle parameter (Tc in the paper), default=3
        2. gamma (float): decay factor (lambda in the paper), default=15
        3. mu0 (float): initial mass parameter (M0 in the paper), default=0.1
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 25,
                 T: int = 3, gamma: float = 15, mu0: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 25
            T (int): cycle parameter (Tc in the paper), default=3
            gamma (float): decay factor (lambda in the paper), default=15
            mu0 (float): initial mass parameter (M0 in the paper), default=0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.T = self.validator.check_int("T", T, [1, 1000])
        self.gamma = self.validator.check_float("gamma", gamma, (0.0, 1000.0))
        self.mu0 = self.validator.check_float("mu0", mu0, (0.0, 10.0))
        self.set_parameters(["epoch", "pop_size", "T", "gamma", "mu0"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        # Orbital eccentricity (Eq. 4) and orbital period (Eq. 5)
        self.orbital = self.generator.random(self.pop_size)
        self.period = np.abs(self.generator.standard_normal(self.pop_size))
        self.velocities = np.zeros((self.pop_size, self.problem.n_dims))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        t_base = (epoch - 1) * self.pop_size
        t_max = self.epoch * self.pop_size
        eps = np.finfo(float).eps

        positions = np.array([agent.solution for agent in self.pop])
        fits = np.array([agent.target.fitness for agent in self.pop])

        # Sun (best-so-far solution)
        sun_pos = self.g_best.solution.copy()
        sun_score = self.g_best.target.fitness

        # Worst fitness in the current population (Eq. 11)
        if self.problem.minmax == "min":
            worst_fitness = np.max(fits)
        else:
            worst_fitness = np.min(fits)

        # Mass parameter (Eq. 12)
        M = self.mu0 * np.exp(-self.gamma * (t_base / t_max))

        # Distance between Sun and planets (Eq. 7)
        R = np.linalg.norm(sun_pos - positions, axis=1)

        # Mass of Sun and planets (Eq. 8, Eq. 9)
        fit_deltas = fits - worst_fitness
        sum_fit = np.sum(fit_deltas)
        if np.isclose(sum_fit, 0.0):
            sum_fit = self.EPSILON
        MS = self.generator.random(self.pop_size) * (sun_score - worst_fitness) / sum_fit
        m = fit_deltas / sum_fit

        # Normalization (Eq. 24)
        r_range = np.ptp(R)
        ms_range = np.ptp(MS)
        m_range = np.ptp(m)
        Rnorm = (R - np.min(R)) / (r_range + eps)
        MSnorm = (MS - np.min(MS)) / (ms_range + eps)
        Mnorm = (m - np.min(m)) / (m_range + eps)

        # Gravitational force (Eq. 6)
        Fg = self.orbital * M * ((MSnorm * Mnorm) / (Rnorm ** 2 + eps)) + self.generator.random(self.pop_size)

        # Semi-major axis (Eq. 23)
        mass_term = M * (MS + m) / (4 * np.pi * np.pi)
        a1 = self.generator.random(self.pop_size) * np.cbrt(self.period ** 2 * mass_term)

        for idx in range(self.pop_size):
            t_current = t_base + idx
            a2 = -1 - (np.remainder(t_current, t_max / self.T) / (t_max / self.T))  # Eq. 29
            n = (a2 - 1) * self.generator.random() + 1  # Eq. 28

            a = self.generator.integers(0, self.pop_size)
            b = self.generator.integers(0, self.pop_size)
            rd = self.generator.random(self.problem.n_dims)
            r = self.generator.random()
            U1 = rd < r  # Eq. 21

            pos_old = positions[idx].copy()

            if self.generator.random() < self.generator.random():
                # Adaptive distance update (Eq. 27, Eq. 26)
                h = 1.0 / np.exp(n * self.generator.standard_normal())  # Eq. 27
                Xm = (positions[b] + sun_pos + pos_old) / 3.0
                mask = U1.astype(float)
                pos_new = pos_old * mask + (Xm + h * (Xm - positions[a])) * (1.0 - mask)  # Eq. 26
            else:
                # Velocity and movement update (Eq. 13-20, Eq. 25)
                f_dir = 1.0 if self.generator.random() < 0.5 else -1.0  # Eq. 18
                L = np.sqrt(M * (MS[idx] + m[idx]) * np.abs((2.0 / (R[idx] + eps)) - (1.0 / (a1[idx] + eps))))  # Eq. 15
                U = rd > self.generator.random(self.problem.n_dims)

                if Rnorm[idx] < 0.5:
                    M_rand = self.generator.random() * (1.0 - r) + r  # Eq. 16
                    l = L * M_rand * U  # Eq. 14
                    Mv = self.generator.random(self.problem.n_dims) * (1.0 - rd) + rd  # Eq. 20
                    l1 = L * Mv * (~U)  # Eq. 19
                    V = (l * (2.0 * self.generator.random() * pos_old - positions[a]) +
                         l1 * (positions[b] - positions[a]) +
                         (1.0 - Rnorm[idx]) * f_dir * U1 *
                         self.generator.random(self.problem.n_dims) * (self.problem.ub - self.problem.lb))  # Eq. 13a
                else:
                    U2 = self.generator.random() > self.generator.random()  # Eq. 22
                    V = (self.generator.random() * L * (positions[a] - pos_old) +
                         (1.0 - Rnorm[idx]) * f_dir * U2 *
                         self.generator.random(self.problem.n_dims) *
                         (self.generator.random() * self.problem.ub - self.problem.lb))  # Eq. 13b

                f_dir = 1.0 if self.generator.random() < 0.5 else -1.0  # Eq. 18
                pos_new = (pos_old + V * f_dir) + (Fg[idx] + np.abs(self.generator.standard_normal())) * U * (sun_pos - pos_old)  # Eq. 25
                self.velocities[idx] = V

            # Boundary handling via Problem.correct_solution()
            pos_new = self.correct_solution(pos_new)

            # Elitism (Eq. 30)
            target_new = self.get_target(pos_new)
            if self.compare_target(target_new, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update_agent(pos_new, target_new)
                positions[idx] = pos_new
                fits[idx] = target_new.fitness
                if self.compare_target(target_new, self.g_best.target, self.problem.minmax):
                    self.g_best = self.pop[idx].copy()
                    sun_pos = self.g_best.solution.copy()
                    sun_score = self.g_best.target.fitness
            else:
                positions[idx] = pos_old
