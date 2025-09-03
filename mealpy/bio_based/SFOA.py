#!/usr/bin/env python
# Created by "Thieu" at 22:37, 03/09/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSFOA(Optimizer):
    """
    The original version: Starfish Optimization Algorithm (SFOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/173735-starfish-optimization-algorithm-sfoa

    Notes:
        This algorithm claims to outperform 95 compared algorithms in accuracy and 97 algorithms in efficiency.
        However, it does not present any remarkable equations. Moreover, the provided MATLAB code does not
        include the standard CEC benchmark functions, but only simplified versions of them.
        Users should carefully consider this when validating the algorithm.
        Many new algorithms claim to be superior to other state-of-the-art methods,
        but it is evident that their implementations are often incorrect.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.5, 3.0] -> better [0.5, 2.0], the greatest step size
        + p_m (float): (0, 1.0) -> better [0.01, 0.2], mutation probability
        + psw (float): (0, 1.0) -> better [0.01, 0.1], proportion of space width (z in the paper)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SFOA
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
    >>> model = SFOA.OriginalSFOA(epoch=1000, pop_size=50, gp = 0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    [1] Zhong, C., Li, G., Meng, Z., Li, H., Yildiz, A. R., & Mirjalili, S. (2025).
    Starfish optimization algorithm (SFOA): a bio-inspired metaheuristic algorithm for global
    optimization compared with 100 optimizers. Neural Computing and Applications, 37(5), 3641-3683.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, gp: float = 0.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            gp (float): the exploration of starfish, default=0.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.gp = self.validator.check_float("gp", gp, [0, 1.0])
        self.set_parameters(["epoch", "pop_size", "gp"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        theta = np.pi / 2 * epoch / self.epoch
        tEO = (self.epoch - epoch) / self.epoch * np.cos(theta)

        pop_new = []
        if self.generator.random() < self.gp:  # exploration of starfish
            for idx in range(self.pop_size):
                pos_new = self.pop[idx].solution.copy()
                if self.problem.n_dims > 5:
                    # for nD is larger than 5
                    jp1 = self.generator.choice(self.problem.n_dims, 5, replace=False)
                    pm = (2 * self.generator.random(size=self.problem.n_dims) - 1) * np.pi
                    pos1 = pos_new + pm * (self.g_best.solution - pos_new) * np.cos(theta)
                    pos2 = pos_new - pm * (self.g_best.solution - pos_new) * np.sin(theta)
                    pos = np.where(self.generator.random(size=self.problem.n_dims) < self.gp, pos1, pos2)
                    pos_new[jp1] = pos[jp1]
                    # Boundary check for individual dimension
                    pos_new[jp1] = np.where((pos_new[jp1] < self.problem.lb[jp1]) | (pos_new[jp1] > self.problem.ub[jp1]),
                                            self.pop[idx].solution[jp1], pos_new[jp1])
                else:
                    # for nD is not larger than 5
                    jp2 = self.generator.integers(0, self.problem.n_dims)
                    im = self.generator.choice(self.pop_size, 2, replace=False)
                    diff1 = self.pop[im[0]].solution[jp2] - pos_new[jp2]
                    diff2 = self.pop[im[1]].solution[jp2] - pos_new[jp2]
                    rand1 = 2 * self.generator.random() - 1
                    rand2 = 2 * self.generator.random() - 1
                    pos_new[jp2] = tEO * pos_new[jp2] + rand1 * diff1 + rand2 * diff2
                    # Boundary check for individual dimension
                    if pos_new[jp2] > self.problem.ub[jp2] or pos_new[jp2] < self.problem.lb[jp2]:
                        pos_new[jp2] = self.pop[idx].solution[jp2]
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
        else:  # exploitation of starfish
            df = self.generator.choice(self.pop_size, 5, replace=False)
            # five arms of starfish
            dm1 = self.g_best.solution - self.pop[df[0]].solution
            dm2 = self.g_best.solution - self.pop[df[1]].solution
            dm3 = self.g_best.solution - self.pop[df[2]].solution
            dm4 = self.g_best.solution - self.pop[df[3]].solution
            dm5 = self.g_best.solution - self.pop[df[4]].solution
            dm = [dm1, dm2, dm3, dm4, dm5]
            for idx in range(self.pop_size):
                r1, r2 = self.generator.random(size=2)
                kp = self.generator.choice(5, size=2, replace=False)
                pos_new = self.pop[idx].solution + r1 * dm[kp[0]] + r2 * dm[kp[1]]  # exploitation
                if idx == self.pop_size - 1:    # last individual
                    pos_new = np.exp(-epoch * self.pop_size / self.epoch) * self.pop[idx].solution  # regeneration of starfish
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
        # Update population with greedy strategy
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
