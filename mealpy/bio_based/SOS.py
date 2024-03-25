#!/usr/bin/env python
# Created by "Thieu" at 14:20, 15/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.optimizer import Optimizer


class OriginalSOS(Optimizer):
    """
    The original version: Symbiotic Organisms Search (SOS)

    Links:
        1. https://doi.org/10.1016/j.compstruc.2014.03.007

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SOS
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
    >>> model = SOS.OriginalSOS(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Cheng, M. Y., & Prayogo, D. (2014). Symbiotic organisms search: a new metaheuristic
    optimization algorithm. Computers & Structures, 139, 98-112.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.is_parallelizable = False
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            ## Mutualism Phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            mutual_vector = (self.pop[idx].solution + self.pop[jdx].solution) / 2
            bf1, bf2 = self.generator.integers(1, 3, 2)
            xi_new = self.pop[idx].solution + self.generator.random() * (self.g_best.solution - bf1 * mutual_vector)
            xj_new = self.pop[jdx].solution + self.generator.random() * (self.g_best.solution - bf2 * mutual_vector)
            xi_new = self.correct_solution(xi_new)
            xj_new = self.correct_solution(xj_new)
            xi_target = self.get_target(xi_new)
            xj_target = self.get_target(xj_new)
            if self.compare_target(xi_target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=xi_new, target=xi_target)
            if self.compare_target(xj_target, self.pop[jdx].target, self.problem.minmax):
                self.pop[jdx].update(solution=xj_new, target=xj_target)
            ## Commensalism phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            xi_new = self.pop[idx].solution + self.generator.uniform(-1, 1) * (self.g_best.solution - self.pop[jdx].solution)
            xi_new = self.correct_solution(xi_new)
            xi_target = self.get_target(xi_new)
            if self.compare_target(xi_target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=xi_new, target=xi_target)
            ## Parasitism phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            temp_idx = self.generator.integers(0, self.problem.n_dims)
            xi_new = self.pop[jdx].solution.copy()
            xi_new[temp_idx] = self.problem.generate_solution()[temp_idx]
            xi_new = self.correct_solution(xi_new)
            xi_target = self.get_target(xi_new)
            if self.compare_target(xi_target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=xi_new, target=xi_target)
