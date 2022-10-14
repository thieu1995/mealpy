#!/usr/bin/env python
# Created by "Thieu" at 07:50, 14/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalRUN(Optimizer):
    """
    The original version of: RUNge Kutta optimizer (RUN)

    Links:
        1. https://doi.org/10.1016/j.eswa.2021.115079
        2. https://imanahmadianfar.com/codes/
        3. https://www.aliasgharheidari.com/RUN.html

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.PSS import OriginalPSS
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = OriginalRUN(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Ahmadianfar, I., Heidari, A. A., Gandomi, A. H., Chu, X., & Chen, H. (2021). RUN beyond the metaphor: An efficient
    optimization algorithm based on Runge Kutta method. Expert Systems with Applications, 181, 115079.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.support_parallel_modes = False
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def runge_kutta__(self, xb, xw, delta_x):
        dim = len(xb)
        C = np.random.randint(1, 3) * (1 - np.random.rand())
        r1 = np.random.random(dim)
        r2 = np.random.random(dim)
        K1 = 0.5 * (np.random.rand() * xw - C * xb)
        K2 = 0.5 * (np.random.rand() * (xw + r2*K1*delta_x/2) - (C*xb + r1*K1*delta_x/2))
        K3 = 0.5 * (np.random.rand() * (xw + r2*K2*delta_x/2) - (C*xb + r1*K2*delta_x/2))
        K4 = 0.5 * (np.random.rand() * (xw + r2*K3*delta_x) - (C*xb + r1*K3*delta_x))
        return (K1 + 2*K2 + 2*K3 + K4)/6

    def uniform_random__(self, a, b, size):
        a2, b2 = a/2, b/2
        mu = a2 + b2
        sig = b2 - a2
        return mu + sig * (2 * np.random.uniform(0, 1, size) - 1)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        f = 20 * np.exp(-(12. * (epoch+1) / self.epoch))        # Eq.17.6
        SF = 2.*(0.5 - np.random.random(self.pop_size)) * f     # Eq.17.5
        x_list = np.array([agent[self.ID_POS] for agent in self.pop])
        x_average = np.mean(x_list, axis=0)     # Determine the Average of Solutions

        nfe_epoch = self.pop_size
        for idx in range(0, self.pop_size):
            ## Determine Delta X (Eqs. 11.1 to 11.3)
            gama = np.random.rand() * (self.pop[idx][self.ID_POS] - np.random.uniform(0, 1, self.problem.n_dims) *
                                       (self.problem.ub - self.problem.lb)) * np.exp(-4*(epoch+1) / self.epoch)
            stp = np.random.uniform(0, 1, self.problem.n_dims) * ((self.g_best[self.ID_POS] - np.random.rand() * x_average) + gama)
            delta_x = 2 * np.random.uniform(0, 1, self.problem.n_dims) * np.abs(stp)

            ## Determine Three Random Indices of Solutions
            a, b, c = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
            id_min_x = self.get_index_best([self.pop[a], self.pop[b], self.pop[c]])
            ## Determine Xb and Xw for using in Runge Kutta method
            if self.compare_agent(self.pop[idx], self.pop[id_min_x]):
                xb, xw = self.pop[idx][self.ID_POS], self.pop[id_min_x][self.ID_POS]
            else:
                xb, xw = self.pop[id_min_x][self.ID_POS], self.pop[idx][self.ID_POS]
            ## Search Mechanism (SM) of RUN based on Runge Kutta Method
            SM = self.runge_kutta__(xb, xw, delta_x)

            _, local_best = self.get_global_best_solution(self.pop)
            L = np.random.choice(range(0, 2), self.problem.n_dims)
            xc = L * self.pop[idx][self.ID_POS] + (1 - L) * self.pop[a][self.ID_POS]        # Eq. 17.3
            xm = L * self.g_best[self.ID_POS] + (1 - L) * local_best[self.ID_POS]           # Eq. 17.4

            r = np.random.choice([1, -1], self.problem.n_dims)          # An Interger number
            g = 2 * np.random.rand()
            mu = 0.5 + 1 * np.random.uniform(0, 1, self.problem.n_dims)

            ## Determine New Solution Based on Runge Kutta Method (Eq.18)
            if np.random.rand() < 0.5:
                pos_new = xc + r * SF[idx] * g * xc + SF[idx] * SM + mu * (xm - xc)
            else:
                pos_new = xm + r * SF[idx] * g * xm + SF[idx] * SM + mu * (self.pop[a][self.ID_POS] - self.pop[b][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            tar_new = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, tar_new], self.pop[idx]):
                self.pop[idx] = [pos_new, tar_new]

            ## Enhanced solution quality (ESQ)  (Eq. 19)
            if np.random.rand() < 0.5:
                w = self.uniform_random__(0, 2, self.problem.n_dims) * np.exp(-5*np.random.rand() * (epoch + 1) / self.epoch)        # Eq.19-1
                r = np.floor(self.uniform_random__(-1, 2, 1))
                u = 2 * np.random.random(self.problem.n_dims)

                a, b, c = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                x_ave = (self.pop[a][self.ID_POS] + self.pop[b][self.ID_POS] + self.pop[c][self.ID_POS]) / 3                # Eq.19-2

                beta = np.random.random(self.problem.n_dims)
                x_new1 = beta * self.g_best[self.ID_POS] + (1 - beta) * x_ave                                               # Eq.19-3

                x_new2_temp1 = x_new1 + r*w * np.abs(np.random.normal(0, 1, self.problem.n_dims) + (x_new1 - x_ave))
                x_new2_temp2 = x_new1 - x_ave + r*w*np.abs(np.random.normal(0, 1, self.problem.n_dims) + u * x_new1 - x_ave)
                x_new2 = np.where(w < 1, x_new2_temp1, x_new2_temp2)
                pos_new2 = self.amend_position(x_new2, self.problem.lb, self.problem.ub)
                tar_new2 = self.get_target_wrapper(pos_new2)
                nfe_epoch += 1

                if self.compare_agent([pos_new2, tar_new2], self.pop[idx]):
                    self.pop[idx] = [pos_new2, tar_new2]
                else:
                    if w[np.random.randint(0, self.problem.n_dims)] > np.random.rand():
                        SM = self.runge_kutta__(self.pop[idx][self.ID_POS], pos_new2, delta_x)
                        x_new3 = pos_new2 - np.random.rand()*pos_new2 + \
                                 SF[idx] * (SM + (2 * np.random.random(self.problem.n_dims)*self.g_best[self.ID_POS] - pos_new2))       # Eq. 20
                        pos_new3 = self.amend_position(x_new3, self.problem.lb, self.problem.ub)
                        tar_new3 = self.get_target_wrapper(pos_new3)
                        nfe_epoch += 1
                        if self.compare_agent([pos_new3, tar_new3], self.pop[idx]):
                            self.pop[idx] = [pos_new3, tar_new3]
        self.nfe_per_epoch = nfe_epoch
