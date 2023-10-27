#!/usr/bin/env python
# Created by "Thieu" at 09:17, 09/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.spatial.distance import cdist
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class DevBRO(Optimizer):
    """
    The developed version: Battle Royale Optimization (BRO)

    Notes:
        + The flow of algorithm is changed. Thrid loop is removed

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + threshold (int): [2, 5], dead threshold, default=3

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BRO
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
    >>> model = BRO.DevBRO(epoch=1000, pop_size=50, threshold = 3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, threshold: float = 3, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            threshold (int): dead threshold, default=3
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.threshold = self.validator.check_float("threshold", threshold, [1, 10])
        self.set_parameters(["epoch", "pop_size", "threshold"])
        self.is_parallelizable = False
        self.sort_flag = False

    def initialize_variables(self):
        shrink = np.ceil(np.log10(self.epoch))
        self.dyn_delta = np.round(self.epoch / shrink)
        self.problem.lb_updated = self.problem.lb.copy()
        self.problem.ub_updated = self.problem.ub.copy()

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        damage = 0
        return Agent(solution=solution, damage=damage)

    def get_idx_min__(self, data):
        k_zero = np.count_nonzero(data == 0)
        if k_zero == len(data):
            return self.generator.choice(range(0, k_zero))
        ## 1st: Partition sorting, not good solution here.
        # return np.argpartition(data, k_zero)[k_zero]
        ## 2nd: Faster
        return np.where(data == np.min(data[data != 0]))[0][0]

    def find_idx_min_distance__(self, target_pos=None, pop=None):
        list_pos = np.array([pop[idx].solution for idx in range(0, self.pop_size)])
        target_pos = np.reshape(target_pos, (1, -1))
        dist_list = cdist(list_pos, target_pos, 'euclidean')
        dist_list = np.reshape(dist_list, (-1))
        return self.get_idx_min__(dist_list)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(self.pop_size):
            # Compare ith soldier with nearest one (jth)
            jdx = self.find_idx_min_distance__(self.pop[idx].solution, self.pop)
            if self.compare_target(self.pop[idx].target, self.pop[jdx].target, self.problem.minmax):
                ## Update Winner based on global best solution
                pos_new = self.pop[idx].solution + self.generator.normal(0, 1) * \
                          np.mean(np.array([self.pop[idx].solution, self.g_best.solution]), axis=0)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                dam_new = self.pop[idx].damage - 1  ## Substract damaged hurt -1 to go next battle
                agent.damage = dam_new
                self.pop[idx] = agent
                ## Update Loser
                if self.pop[jdx].damage < self.threshold:  ## If loser not dead yet, move it based on general
                    pos_new = self.generator.uniform() * (np.maximum(self.pop[jdx].solution, self.g_best.solution) -
                            np.minimum(self.pop[jdx].solution, self.g_best.solution)) + np.maximum(self.pop[jdx].solution, self.g_best.solution)
                    dam_new = self.pop[jdx].damage + 1
                    self.pop[jdx].target = self.get_target(self.pop[jdx].solution)
                else:  ## Loser dead and respawn again
                    pos_new = self.generator.uniform(self.problem.lb_updated, self.problem.ub_updated)
                    dam_new = 0
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                agent.damage = dam_new
                self.pop[jdx] = agent
            else:
                ## Update Loser by following position of Winner
                self.pop[idx] = self.pop[jdx].copy()
                ## Update Winner by following position of General to protect the King and General
                pos_new = self.pop[jdx].solution + self.generator.uniform() * (self.g_best.solution - self.pop[jdx].solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                agent.damage = 0
                self.pop[jdx] = agent
        if epoch >= self.dyn_delta:  # max_epoch = 1000 -> delta = 300, 450, >500,....
            pos_list = np.array([self.pop[idx].solution for idx in range(0, self.pop_size)])
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best.solution - pos_std
            ub = self.g_best.solution + pos_std
            self.problem.lb_updated = np.clip(lb, self.problem.lb_updated, self.problem.ub_updated)
            self.problem.ub_updated = np.clip(ub, self.problem.lb_updated, self.problem.ub_updated)
            self.dyn_delta += np.round(self.dyn_delta / 2)


class OriginalBRO(DevBRO):
    """
    The original version of: Battle Royale Optimization (BRO)

    Links:
        1. https://doi.org/10.1007/s00521-020-05004-4

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + threshold (int): [2, 5], dead threshold, default=3

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BRO
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
    >>> model = BRO.OriginalBRO(epoch=1000, pop_size=50, threshold = 3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rahkar Farshi, T., 2021. Battle royale optimization algorithm. Neural Computing and Applications, 33(4), pp.1139-1157.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, threshold: float = 3, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            threshold (int): dead threshold, default=3
        """
        super().__init__(epoch, pop_size, threshold, **kwargs)
        self.is_parallelizable = False
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(self.pop_size):
            # Compare ith soldier with nearest one (jth)
            jdx = self.find_idx_min_distance__(self.pop[idx].solution, self.pop)
            dam, vic = idx, jdx  ## This error in the algorithm's flow in the paper, But in the matlab code, he changed.
            if self.compare_target(self.pop[idx].target, self.pop[jdx].target, self.problem.minmax):
                dam, vic = jdx, idx  ## The mistake also here in the paper.
            if self.pop[dam].damage < self.threshold:
                pos_new = self.generator.uniform(0, 1, self.problem.n_dims) * (np.maximum(self.pop[dam].solution, self.g_best.solution) -
                        np.minimum(self.pop[dam].solution, self.g_best.solution)) + np.maximum(self.pop[dam].solution, self.g_best.solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                agent.damage = self.pop[dam].damage + 1
                self.pop[dam] = agent
                self.pop[vic].damage = 0
            else:
                pos_new = self.generator.uniform(self.problem.lb_updated, self.problem.ub_updated)
                agent = self.generate_agent(pos_new)
                self.pop[dam] = agent
        if epoch >= self.dyn_delta:
            pos_list = np.array([self.pop[idx].solution for idx in range(0, self.pop_size)])
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best.solution - pos_std
            ub = self.g_best.solution + pos_std
            self.problem.lb_updated = np.clip(lb, self.problem.lb_updated, self.problem.ub_updated)
            self.problem.ub_updated = np.clip(ub, self.problem.lb_updated, self.problem.ub_updated)
            self.dyn_delta += round(self.dyn_delta / 2)
