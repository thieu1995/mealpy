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
    >>> from mealpy import FloatVar, RUN
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = RUN.OriginalRUN(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ahmadianfar, I., Heidari, A. A., Gandomi, A. H., Chu, X., & Chen, H. (2021). RUN beyond the metaphor: An efficient
    optimization algorithm based on Runge Kutta method. Expert Systems with Applications, 181, 115079.
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
        self.is_parallelizable = False
        self.sort_flag = False

    def runge_kutta__(self, xb, xw, delta_x):
        dim = len(xb)
        C = self.generator.integers(1, 3) * (1 - self.generator.random())
        r1 = self.generator.random(dim)
        r2 = self.generator.random(dim)
        K1 = 0.5 * (self.generator.random() * xw - C * xb)
        K2 = 0.5 * (self.generator.random() * (xw + r2*K1*delta_x/2) - (C*xb + r1*K1*delta_x/2))
        K3 = 0.5 * (self.generator.random() * (xw + r2*K2*delta_x/2) - (C*xb + r1*K2*delta_x/2))
        K4 = 0.5 * (self.generator.random() * (xw + r2*K3*delta_x) - (C*xb + r1*K3*delta_x))
        return (K1 + 2*K2 + 2*K3 + K4)/6

    def uniform_random__(self, a, b, size):
        a2, b2 = a/2, b/2
        mu = a2 + b2
        sig = b2 - a2
        return mu + sig * (2 * self.generator.uniform(0, 1, size) - 1)

    def get_index_of_best_agent__(self, pop):
        fit_list = np.array([agent.target.fitness for agent in pop])
        if self.problem.minmax == "min":
            return np.argmin(fit_list)
        else:
            return np.argmax(fit_list)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        f = 20 * np.exp(-(12. * epoch / self.epoch))        # Eq.17.6
        SF = 2.*(0.5 - self.generator.random(self.pop_size)) * f     # Eq.17.5
        x_list = np.array([agent.solution for agent in self.pop])
        x_average = np.mean(x_list, axis=0)     # Determine the Average of Solutions
        for idx in range(0, self.pop_size):
            ## Determine Delta X (Eqs. 11.1 to 11.3)
            gama = self.generator.random() * (self.pop[idx].solution - self.generator.uniform(0, 1, self.problem.n_dims) *
                                       (self.problem.ub - self.problem.lb)) * np.exp(-4 * epoch / self.epoch)
            stp = self.generator.uniform(0, 1, self.problem.n_dims) * ((self.g_best.solution - self.generator.random() * x_average) + gama)
            delta_x = 2 * self.generator.uniform(0, 1, self.problem.n_dims) * np.abs(stp)
            ## Determine Three Random Indices of Solutions
            a, b, c = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
            id_min_x = self.get_index_of_best_agent__([self.pop[a], self.pop[b], self.pop[c]])
            ## Determine Xb and Xw for using in Runge Kutta method
            if self.compare_target(self.pop[idx].target, self.pop[id_min_x].target, self.problem.minmax):
                xb, xw = self.pop[idx].solution, self.pop[id_min_x].solution
            else:
                xb, xw = self.pop[id_min_x].solution, self.pop[idx].solution
            ## Search Mechanism (SM) of RUN based on Runge Kutta Method
            SM = self.runge_kutta__(xb, xw, delta_x)
            local_best = self.get_best_agent(self.pop, self.problem.minmax)
            L = self.generator.choice(range(0, 2), self.problem.n_dims)
            xc = L * self.pop[idx].solution + (1 - L) * self.pop[a].solution        # Eq. 17.3
            xm = L * self.g_best.solution + (1 - L) * local_best.solution           # Eq. 17.4
            r = self.generator.choice([1, -1], self.problem.n_dims)          # An integer number
            g = 2 * self.generator.random()
            mu = 0.5 + 1 * self.generator.uniform(0, 1, self.problem.n_dims)
            ## Determine New Solution Based on Runge Kutta Method (Eq.18)
            if self.generator.random() < 0.5:
                pos_new = xc + r * SF[idx] * g * xc + SF[idx] * SM + mu * (xm - xc)
            else:
                pos_new = xm + r * SF[idx] * g * xm + SF[idx] * SM + mu * (self.pop[a].solution - self.pop[b].solution)
            pos_new = self.correct_solution(pos_new)
            tar_new = self.get_target(pos_new)
            if self.compare_target(tar_new, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new, target=tar_new)
            ## Enhanced solution quality (ESQ)  (Eq. 19)
            if self.generator.random() < 0.5:
                w = self.uniform_random__(0, 2, self.problem.n_dims) * np.exp(-5*self.generator.random() * epoch / self.epoch)        # Eq.19-1
                r = np.floor(self.uniform_random__(-1, 2, 1))
                u = 2 * self.generator.random(self.problem.n_dims)
                a, b, c = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                x_ave = (self.pop[a].solution + self.pop[b].solution + self.pop[c].solution) / 3                # Eq.19-2
                beta = self.generator.random(self.problem.n_dims)
                x_new1 = beta * self.g_best.solution + (1 - beta) * x_ave                                               # Eq.19-3
                x_new2_temp1 = x_new1 + r*w * np.abs(self.generator.normal(0, 1, self.problem.n_dims) + (x_new1 - x_ave))
                x_new2_temp2 = x_new1 - x_ave + r*w*np.abs(self.generator.normal(0, 1, self.problem.n_dims) + u * x_new1 - x_ave)
                x_new2 = np.where(w < 1, x_new2_temp1, x_new2_temp2)
                pos_new2 = self.correct_solution(x_new2)
                tar_new2 = self.get_target(pos_new2)
                if self.compare_target(tar_new2, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx].update(solution=pos_new2, target=tar_new2)
                else:
                    if w[self.generator.integers(0, self.problem.n_dims)] > self.generator.random():
                        SM = self.runge_kutta__(self.pop[idx].solution, pos_new2, delta_x)
                        x_new3 = pos_new2 - self.generator.random()*pos_new2 + \
                                 SF[idx] * (SM + (2 * self.generator.random(self.problem.n_dims)*self.g_best.solution - pos_new2))       # Eq. 20
                        pos_new3 = self.correct_solution(x_new3)
                        tar_new3 = self.get_target(pos_new3)
                        if self.compare_target(tar_new3, self.pop[idx].target, self.problem.minmax):
                            self.pop[idx].update(solution=pos_new3, target=tar_new3)
