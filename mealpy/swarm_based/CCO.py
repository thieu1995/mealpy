#!/usr/bin/env python
# Created by "Karahan Ballı" on 07/01/2026
# Github: https://github.com/karahanballi
# ---------------------------------------%
# Updated by "Thieu" on 12/07/2026
# Github: https://github.com/thieu1995
# ---------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalCCO(Optimizer):
    """
    The original version of: Cuckoo Catfish Optimizer (CCO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.
    alpha : float
        Good range (0, 10.0), alpha parameter, default = 1.34 (as in Matlab code).
    beta : float
        Good range (0, 10.0), beta parameter, default = 0.3 (as in Matlab code).

    Danger
    ------
    1. Excessive Complexity: This is the most complex and convoluted algorithm we have ever implemented.
       It is packed with nested operators that seem entirely disconnected from the actual behavior of a Cuckoo Catfish.
    2. Arbitrary Logic: We get the distinct impression that the authors simply fabricated the equations and
       an excessive number of if-else conditions just to force the algorithm to perform well.
    3. Lack of Conceptual Alignment: While the algorithm may appear mathematically sound, it lacks any real
       connection to its stated inspiration, the Cuckoo Catfish. We would advise researchers, especially
       those new to the field to avoid using such overly complex heuristics for development.
    4. Computational Inefficiency: A major issue is that the actual computational complexity does not
       align with the claims made in the paper. The excessive sorting processes within the
       population update loops make the algorithm significantly slower than others.
    5. Lack of Parallelizability: Furthermore, the algorithm is not inherently parallelizable,
       as the population updates are strictly interdependent.

    Links
    -----
    1. https://www.mathworks.com/matlabcentral/fileexchange/176828-cuckoo-catfish-optimizer-a-new-meta-heuristic-optimization
    2. https://doi.org/10.1007/s10462-025-11291-x

    References
    ----------
    1. Wang, T. L., Gu, S. W., Liu, R. J., Chen, L. Q., Wang, Z., & Zeng, Z. Q. (2025).
       Cuckoo catfish optimizer: a new meta-heuristic optimization algorithm. Artificial Intelligence Review, 58(10), 326.

    Examples
    --------
    ::

        import numpy as np
        from mealpy import FloatVar, CCO

        def objective_function(solution):
            return np.sum(solution**2)

        problem_dict = {
            "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="x"),
            "minmax": "min",
            "obj_func": objective_function,
        }

        model = CCO.OriginalCCO(epoch=1000, pop_size=50, alpha=0.5, beta=1.0)
        g_best = model.solve(problem_dict)
        print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Cuckoo Catfish Optimizer", year=2025, difficulty="hard", kind="original",
                       scientific_status="questionable",
                       concerns=(
                           ScientificConcern.LACK_OF_NOVELTY, ScientificConcern.QUESTIONABLE_MATH,
                           ScientificConcern.POOR_REPRODUCIBILITY, ScientificConcern.AMBIGUOUS_METHODOLOGY
                       ))

    def __init__(self, epoch=10000, pop_size=100, alpha=1.34, beta=0.3, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 1000000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 100000])
        self.alpha = self.validator.check_float("alpha", alpha, (0, 10.0))
        self.beta = self.validator.check_float("beta", beta, (0, 10.0))
        self.set_parameters(["epoch", "pop_size", "alpha", "beta"])
        self.sort_flag = False
        self.is_parallelizable = False

    def before_main_loop(self):
        """
        Initialize algorithm-specific variables based on the MATLAB initialization block.
        """
        self.x = np.zeros(self.pop_size)
        self.y = np.zeros(self.pop_size)

        # Reproducing MATLAB's theta, x, and y initialization
        for i in range(self.pop_size):
            # i+1 to match MATLAB's 1-based index calculation
            theta = (1 - 10 * (i + 1) / self.pop_size) * np.pi
            r = self.alpha * np.exp(self.beta * theta / 3)
            self.x[i] = r * np.cos(theta)
            self.y[i] = r * np.sin(theta)

        self.s = 0
        self.z = 0
        self.tt = 0

        # Calculate initial Dis and Lx
        pop_pos = np.array([ag.solution for ag in self.pop])

        # Dis = abs(mean(mean((PopPos-BestX)./(WorstX-BestX+eps))))
        diff = self.g_worst.solution - self.g_best.solution + self.EPSILON
        self.Dis = np.abs(np.mean((pop_pos - self.g_best.solution) / diff))
        self.Lx = np.abs(self.generator.standard_normal()) * self.generator.random()

    def evolve(self, epoch):
        """
        The main evolution step called by the Mealpy framework.
        """
        # Time-dependent parameters
        C = 1.0 - (epoch / self.epoch)
        T = (1.0 - np.sin((np.pi * epoch) / (2.0 * self.epoch))) ** (epoch / self.epoch)

        # Dynamic probability for catfish mechanism (die probability)
        if self.tt < 15:
            die = 0.02 * T
        else:
            die = 0.02
            C = 0.8

        _, _, worsts = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        worst = worsts[0].solution
        best = self.g_best.solution

        n_val = 1
        rng = self.generator
        for idx in range(self.pop_size):
            pos_ii = self.pop[idx].solution
            # Stochastic switch
            Q = 0
            F = np.sign(0.5 - rng.random())
            E = n_val * T + rng.random()
            R1 = rng.random(self.problem.n_dims)
            R4 = rng.random(self.problem.n_dims)
            r1 = rng.random()
            r2 = rng.random()
            S = np.sin(np.pi * R4 * C)
            kk = rng.permutation(self.pop_size)

            if rng.random() > C:
                diff = worst - best + self.EPSILON
                J_i = np.abs(np.mean((pos_ii - best) / diff))

                if rng.random() > C:
                    Cy = 1 / (np.pi * (1 + C ** 2))
                    if J_i > self.Dis:
                        pos_new = best + F * S * (best - pos_ii)
                    else:
                        if self.Dis * self.Lx < J_i:
                            pos_new = best * (1 + (T ** 5) * Cy * E) + F * (S * (best - pos_ii))
                        else:
                            pos_new = best * (1 + (T ** 5) * rng.normal(0, C ** 2)) + F * S * (best - pos_ii)
                else:
                    if rng.random() > C:
                        if (idx + 1) % 2 == 1:
                            r3 = rng.random()
                            step = best - E * pos_ii
                            pos_new = ((C / epoch) * (r1 * best - r3 * pos_ii) +
                                       (T ** 2) * self.get_levy_flight_step(1.5, multiplier=0.05, size=self.problem.n_dims, case=-1) * np.abs(step))
                        else:
                            R2 = rng.random(self.problem.n_dims)
                            R3 = rng.random(self.problem.n_dims)
                            step = pos_ii - E * best
                            DE = C * F
                            pos_new = 0.5 * (best + self.pop[kk[0]].solution) + DE * (2 * R1 * step - (R2 / 2) * (DE * R3 - 1))
                    else:
                        if rng.random() < rng.random():
                            if J_i < self.Dis:
                                mean_pop = np.mean([ag.solution for ag in self.pop], axis=0)
                                V = 2 * (rng.random() * (mean_pop - pos_ii) + rng.random() * (best - pos_ii))
                            else:
                                V = 2 * (rng.random() * (self.pop[kk[1]].solution - self.pop[kk[2]].solution) + rng.random() * (self.pop[kk[0]].solution - pos_ii))

                            if self.compare_target(self.pop[idx].target, self.pop[kk[idx]].target):
                                step = pos_ii - E * self.pop[kk[idx]].solution
                                if (idx + 1) % 2 == 1:
                                    pos_new = (pos_ii + (T ** 2) * self.y[idx] * (1 - R1) * np.abs(step)) + F * R1 * step / 2 + V * J_i / epoch
                                else:
                                    pos_new = (pos_ii + (T ** 2) * self.x[idx] * (1 - R1) * np.abs(step)) + F * R1 * step / 2 + V * J_i / epoch
                            else:
                                step = self.pop[kk[idx]].solution - E * pos_ii
                                if (idx + 1) % 2 == 1:
                                    pos_new = (self.pop[kk[idx]].solution + (T ** 2) * self.y[idx] * (1 - R1) * np.abs(step)) + F * R1 * step / 2 + V * J_i / epoch
                                else:
                                    pos_new = (self.pop[kk[idx]].solution + (T ** 2) * self.x[idx] * (1 - R1) * np.abs(step)) + F * R1 * step / 2 + V * J_i / epoch

                            self.s += 1
                            if self.s > 10:
                                rp1, rp2 = rng.choice(self.pop_size, 2, replace=False)
                                lesp1 = r1 * self.pop[rp1].solution + (1 - r1) * self.pop[rp2].solution
                                pos_new = np.round(lesp1) + F * r1 * R1 / (epoch ** 4) * pos_new
                                self.s = 0
                        else:
                            pop_sorted, _ = self.get_sorted_population(self.pop, self.problem.minmax)
                            A2 = rng.integers(4)
                            if A2 == 3:
                                Q = 1
                            A1 = rng.integers(4)
                            mean_pop = np.mean([ag.solution for ag in self.pop], axis=0)
                            D = np.array([pop_sorted[0].solution, pop_sorted[1].solution, pop_sorted[2].solution, mean_pop])
                            B = D[A1]
                            Rt1 = rng.choice(np.arange(1, 361), self.problem.n_dims, replace=True) * np.pi / 360
                            Rt2 = rng.choice(np.arange(1, 361), self.problem.n_dims, replace=True) * np.pi / 360
                            w = 1 - ((np.exp(epoch / self.epoch) - 1) / (np.exp(1) - 1)) ** 2

                            rand_val = rng.random()
                            if rand_val < 0.33:
                                pos_new = B + 2 * w * F * np.cos(Rt1) * np.sin(Rt2) * (B - pos_ii)
                            elif rand_val < 0.66:
                                pos_new = B + 2 * w * F * np.sin(Rt1) * np.cos(Rt2) * (B - pos_ii)
                            else:
                                pos_new = B + 2 * w * F * np.cos(Rt2) * (B - pos_ii)

                            self.z += 1
                            if self.z > 5:
                                rp3 = rng.choice(self.pop_size)
                                pos_new = best * (1.0 - (1.0 - 1.0 / (self.pop[rp3].solution + self.EPSILON)) * R1)
                                self.z = 0
            else:
                if rng.random() > C:
                    if rng.random() > C:
                        pos_new = self.pop[kk[2]].solution + np.abs(rng.normal()) * (best - pos_ii + self.pop[kk[0]].solution - self.pop[kk[1]].solution)
                    else:
                        Z2 = rng.random(self.problem.n_dims) < rng.random()
                        pos_new = Z2 * (self.pop[kk[2]].solution + np.abs(rng.normal()) * (self.pop[kk[0]].solution - self.pop[kk[1]].solution)) + (~Z2) * pos_ii
                else:
                    Z1 = int(rng.random() < rng.random())
                    pos_new = pos_ii + (Z1 * np.abs(rng.normal()) * ((best + self.pop[kk[0]].solution) / 2 - self.pop[kk[1]].solution) +
                                        rng.random() / 2 * (self.pop[kk[2]].solution - self.pop[kk[3]].solution))

                if rng.random() > C or self.tt > 0.8 * self.pop_size:
                    mask = rng.random(self.problem.n_dims) < (0.2 * C + 0.2)
                    pos_new = np.where(mask, pos_new, pos_ii)

            # Die condition block
            if rng.random() < die:
                if rng.random() > C:
                    pos_new = rng.random() * (self.problem.ub - self.problem.lb) + self.problem.lb
                else:
                    term1 = self.get_levy_flight_step(beta=1.5, multiplier=0.05, case=-1) * int(r1 > r2)
                    term2 = np.abs(rng.normal()) * int(r1 <= r2)
                    best_vec = best * (term1 + term2)
                    Upc = np.max(best_vec)
                    Lowc = np.min(best_vec)
                    pos_new = np.ones(self.problem.n_dims) * (rng.random() * (Upc - Lowc) + Lowc)

            # Boundary control
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            # Replacement logic
            if self.compare_target(agent.target, self.pop[idx].target):
                self.pop[idx] = agent
                # Q=1 logic: Replace the current worst in population
                if Q == 1:
                    _, worst_idx = self.get_worst_agent(self.pop, self.problem.minmax)
                    self.pop[worst_idx] = agent.copy()
                self.tt = 0
            else:
                self.tt += 1

            # Update Best and worst sequentially
            _, bests, worsts = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
            best = bests[0].solution
            worst = worsts[0].solution
