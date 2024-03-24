#!/usr/bin/env python
# Created by "Thieu" at 17:29, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalINFO(Optimizer):
    """
    The original version of: weIghted meaN oF vectOrs (INFO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0957417422000173
        2. https://aliasgharheidari.com/INFO.html
        3. https://doi.org/10.1016/j.eswa.2022.116516

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, INFO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = INFO.OriginalINFO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ahmadianfar, I., Heidari, A. A., Noshadian, S., Chen, H., & Gandomi, A. H. (2022). INFO: An efficient optimization
    algorithm based on weighted mean of vectors. Expert Systems with Applications, 195, 116516.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        alpha = 2 * np.exp(-4 * (epoch / self.epoch))      # Eqs.(5.1) - Eq.(9.1)
        idx_better = self.generator.integers(2, 6)
        better = self.pop[idx_better]
        g_worst = self.pop[-1]

        pop_new = []
        for idx in range(0, self.pop_size):
            ## Updating rule stage
            delta = 2 * self.generator.random() * alpha - alpha            # Eq. (5)
            sigma = 2 * self.generator.random() * alpha - alpha            # Eq. (9)
            ## Select three random solution
            a, b, c = self.generator.choice(range(0, self.pop_size), 3, replace=False)
            e1 = 1e-25
            epsilon = e1 * self.generator.random()
            fit_a = self.pop[a].target.fitness
            fit_b = self.pop[b].target.fitness
            fit_c = self.pop[c].target.fitness
            omg1 = np.max([fit_a, fit_b, fit_c])
            MM1 = np.array([fit_a - fit_b, fit_a - fit_c, fit_b - fit_c])

            w1 = np.cos(MM1[0] + np.pi) * np.exp(-np.abs(MM1[0] / omg1))       # Eq. (4.2)
            w2 = np.cos(MM1[1] + np.pi) * np.exp(-np.abs(MM1[1] / omg1))       # Eq. (4.3)
            w3 = np.cos(MM1[2] + np.pi) * np.exp(-np.abs(MM1[2] / omg1))       # Eq. (4.4)
            Wt1 = np.sum([w1, w2, w3])
            WM1 = delta * (w1 * (self.pop[a].solution - self.pop[b].solution) +         #  Eq.(4.1)
                w2 * (self.pop[a].solution - self.pop[c].solution) +
                w3 * (self.pop[b].solution - self.pop[c].solution)) / (Wt1 + 1) + epsilon

            fit_1 = self.g_best.target.fitness
            fit_2 = better.target.fitness
            fit_3 = g_worst.target.fitness
            omg2 = np.max([fit_1, fit_2, fit_3])
            MM2 = np.array([fit_1 - fit_2, fit_1 - fit_3, fit_2 - fit_3])
            w4 = np.cos(MM2[0] + np.pi) * np.exp(-np.abs(MM2[0] / omg2))        # Eq. (4.7)
            w5 = np.cos(MM2[1] + np.pi) * np.exp(-np.abs(MM2[1] / omg2))        # Eq. (4.8)
            w6 = np.cos(MM2[2] + np.pi) * np.exp(-np.abs(MM2[2] / omg2))        # Eq. (4.9)
            Wt2 = np.sum([w4, w5, w6])
            WM2 = delta * (w4 * (self.g_best.solution - better.solution) +          # Eq. (4.6)
                           w5 * (self.g_best.solution - g_worst.solution) +
                            w6 * (better.solution - g_worst.solution)) / (Wt2 + 1) + epsilon
            ## Determine MeanRule
            r = self.generator.uniform(0.1, 0.5)
            mean_rule = r * WM1 + (1 - r) * WM2         # Eq. (4)
            if self.generator.random() < 0.5:                # Eq. (8)
                z1 = self.pop[idx].solution + sigma * (self.generator.random() * mean_rule) + self.generator.random() * \
                     (self.g_best.solution - self.pop[a].solution) / (fit_1 - fit_a + 1)
                z2 = self.g_best.solution + sigma * (self.generator.random() * mean_rule) + self.generator.random() * \
                     (self.pop[a].solution - self.pop[b].solution) / (fit_a - fit_b + 1)
            else:
                z1 = self.pop[a].solution + sigma * (self.generator.random() * mean_rule) + self.generator.random() * \
                     (self.pop[b].solution - self.pop[c].solution) / (fit_b - fit_c + 1)
                z2 = better.solution + sigma * (self.generator.random() * mean_rule) + self.generator.random() * \
                     (self.pop[a].solution - self.pop[b].solution) / (fit_a - fit_b + 1)
            ## Vector combining stage
            mu = 0.05 * self.generator.random(self.problem.n_dims)
            u1 = z1 + mu * np.abs(z1 - z2)      # Eq. (10.1)
            u2 = z2 + mu * np.abs(z1 - z2)      # Eq. (10.2)
            cond1 = self.generator.random(self.problem.n_dims) < 0.05
            cond2 = self.generator.random(self.problem.n_dims) < 0.05
            x1 = np.where(cond1, u1, u2)
            pos_new = np.where(cond2, x1, self.pop[idx].solution)       # Eq. (10.3)
            ## Local search stage
            if self.generator.random() < 0.5:
                L = int(self.generator.random() < 0.5)     # 0 or 1
                v1 = (1 - L) * 2 * self.generator.random() + L     # Eqs. (11.5)
                v2 = self.generator.random() * L + (1 - L)         # Eq. (11.6)
                x_avg = (self.pop[a].solution + self.pop[b].solution + self.pop[c].solution) / 3            # Eq. (11.4)
                phi = self.generator.random()
                x_rand = phi * x_avg + (1 - phi) * (phi * better.solution + (1 - phi) * self.g_best.solution)   # Eq. (11.3)
                n_rand = L * self.generator.random(self.problem.n_dims) + (1 - L) * self.generator.random()
                if self.generator.random() < 0.5:          # Eq. (11.1)
                    pos_new = self.g_best.solution + n_rand * (mean_rule + self.generator.random() * (self.g_best.solution - self.pop[a].solution))
                else:                               # Eq. (11.2)
                    pos_new = x_rand + n_rand * (mean_rule + self.generator.random() * (v1 * self.g_best.solution - v2 * x_rand))
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)
