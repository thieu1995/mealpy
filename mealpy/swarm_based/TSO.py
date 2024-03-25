#!/usr/bin/env python
# Created by "Thieu" at 17:52, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalTSO(Optimizer):
    """
    The original version of: Tuna Swarm Optimization (TSO)

    Notes:
        1. Two variables that authors consider it as a constants (aa = 0.7 and zz = 0.05)
        2. https://www.hindawi.com/journals/cin/2021/9210050/
        3. https://www.mathworks.com/matlabcentral/fileexchange/101734-tuna-swarm-optimization

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TSO
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
    >>> model = TSO.OriginalTSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Xie, L., Han, T., Zhou, H., Zhang, Z. R., Han, B., & Tang, A. (2021). Tuna swarm optimization: a novel swarm-based
    metaheuristic algorithm for global optimization. Computational intelligence and Neuroscience, 2021.
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

    def initialize_variables(self):
        self.aa = 0.7
        self.zz = 0.05

    def get_new_local_pos__(self, C, a1, a2, t, epoch):
        if self.generator.random() < self.zz:
            local_pos = self.problem.generate_solution()
        else:
            if self.generator.random() < 0.5:
                r1 = self.generator.random()
                beta = np.exp(r1 * np.exp(3*np.cos(np.pi*((self.epoch - epoch) / self.epoch)))) * np.cos(2*np.pi*r1)
                if self.generator.random() < C:
                    local_pos = a1*(self.g_best.solution + beta * np.abs(self.g_best.solution - self.pop[0].solution)) + a2 * self.pop[0].solution   # Eq (8.3)
                else:
                    rand_pos = self.problem.generate_solution()
                    local_pos = a1 * (rand_pos + beta*np.abs(rand_pos - self.pop[0].solution)) + a2 * self.pop[0].solution  # Eq (8.1)
            else:
                tf = self.generator.choice([-1, 1])
                if self.generator.random() < 0.5:
                    local_pos = tf * t**2 * self.pop[0].solution        # Eq 9.2
                else:
                    local_pos = self.g_best.solution + self.generator.random(self.problem.n_dims) * (self.g_best.solution - self.pop[0].solution) + \
                        tf * t**2 * (self.g_best.solution - self.pop[0].solution)
        return local_pos

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        C = epoch / self.epoch
        a1 = self.aa + (1 - self.aa) * C
        a2 = (1 - self.aa) - (1 - self.aa) * C
        tt = (1 - epoch / self.epoch) ** (epoch / self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx == 0:
                pos_new = self.get_new_local_pos__(C, a1, a2, tt, epoch)
            else:
                if self.generator.random() < self.zz:
                    pos_new = self.problem.generate_solution()
                else:
                    if self.generator.random() > 0.5:
                        r1 = self.generator.random()
                        beta = np.exp(r1 * np.exp(3*np.cos(np.pi * (self.epoch - epoch)/self.epoch))) * np.cos(2*np.pi*r1)
                        if self.generator.random() < C:
                            pos_new = a1 * (self.g_best.solution + beta*np.abs(self.g_best.solution - self.pop[idx].solution)) + \
                                a2 * self.pop[idx-1].solution       # Eq. 8.4
                        else:
                            rand_pos = self.problem.generate_solution()
                            pos_new = a1 * (rand_pos + beta*np.abs(rand_pos - self.pop[idx].solution)) + a2 * self.pop[idx-1].solution  # Eq 8.2
                    else:
                        tf = self.generator.choice([-1, 1])
                        if self.generator.random() < 0.5:
                            pos_new = self.g_best.solution + self.generator.random(self.problem.n_dims) * \
                                      (self.g_best.solution - self.pop[idx].solution) + tf * tt**2 * (self.g_best.solution - self.pop[idx].solution) # Eq 9.1
                        else:
                            pos_new = tf * tt**2 * self.pop[idx].solution        # Eq 9.2
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)
