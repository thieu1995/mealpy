#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevMVO(Optimizer):
    """
    The developed version: Multi-Verse Optimizer (MVO)

    Notes:
        + New routtele wheel selection can handle negative values
        + Removed condition when self.generator.normalize fitness. So the chance to choose while whole higher --> better

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wep_min (float): [0.05, 0.3], Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
        + wep_max (float: [0.75, 1.0], Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MVO
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
    >>> model = MVO.DevMVO(epoch=1000, pop_size=50, wep_min = 0.2, wep_max = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, wep_min: float = 0.2, wep_max: float = 1.0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wep_min (float): Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
            wep_max (float: Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.wep_min = self.validator.check_float("wep_min", wep_min, (0, 0.5))
        self.wep_max = self.validator.check_float("wep_max", wep_max, [0.5, 3.0])
        self.set_parameters(["epoch", "pop_size", "wep_min", "wep_max"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Eq. (3.3) in the paper
        wep = self.wep_max - epoch * ((self.wep_max - self.wep_min) / self.epoch)
        # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        tdr = 1 - epoch ** (1.0 / 6) / self.epoch ** (1.0 / 6)
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.uniform() < wep:
                list_fitness = np.array([agent.target.fitness for agent in self.pop])
                white_hole_id = self.get_index_roulette_wheel_selection(list_fitness)
                black_hole_pos_1 = self.pop[idx].solution + tdr * self.generator.normal(0, 1) * \
                                   (self.pop[white_hole_id].solution - self.pop[idx].solution)
                black_hole_pos_2 = self.g_best.solution + tdr * self.generator.normal(0, 1) * (self.g_best.solution - self.pop[idx].solution)
                black_hole_pos = np.where(self.generator.random(self.problem.n_dims) < 0.5, black_hole_pos_1, black_hole_pos_2)
            else:
                black_hole_pos = self.problem.generate_solution()
            pos_new = self.correct_solution(black_hole_pos)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class OriginalMVO(DevMVO):
    """
    The original version of: Multi-Verse Optimizer (MVO)

    Links:
        1. https://dx.doi.org/10.1007/s00521-015-1870-7
        2. https://www.mathworks.com/matlabcentral/fileexchange/50112-multi-verse-optimizer-mvo

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wep_min (float): [0.05, 0.3], Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
        + wep_max (float: [0.75, 1.0], Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MVO
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
    >>> model = MVO.OriginalMVO(epoch=1000, pop_size=50, wep_min = 0.2, wep_max = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Mirjalili, S.M. and Hatamlou, A., 2016. Multi-verse optimizer: a nature-inspired
    algorithm for global optimization. Neural Computing and Applications, 27(2), pp.495-513.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, wep_min: float = 0.2, wep_max: float = 1.0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wep_min (float): Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
            wep_max (float: Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
        """
        super().__init__(epoch, pop_size, wep_min, wep_max, **kwargs)

    # sorted_inflation_rates
    def roulette_wheel_selection__(self, weights=None):
        accumulation = np.cumsum(weights)
        p = self.generator.uniform() * accumulation[-1]
        chosen_idx = None
        for idx in range(len(accumulation)):
            if accumulation[idx] > p:
                chosen_idx = idx
                break
        return chosen_idx

    def normalize__(self, d, to_sum=True):
        # d is a (n x dimension) np np.array
        d -= np.min(d, axis=0)
        if to_sum:
            total_vector = np.sum(d, axis=0)
            if 0 in total_vector:
                return self.generator.uniform(0.2, 0.8, self.pop_size)
            return d / np.sum(d, axis=0)
        else:
            ptp_vector = np.ptp(d, axis=0)
            if 0 in ptp_vector:
                return self.generator.uniform(0.2, 0.8, self.pop_size)
            return d / np.ptp(d, axis=0)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Eq. (3.3) in the paper
        wep = self.wep_min + epoch * ((self.wep_max - self.wep_min) / self.epoch)
        # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        tdr = 1 - epoch ** (1.0 / 6) / self.epoch ** (1.0 / 6)
        list_fitness_raw = np.array([item.target.fitness for item in self.pop])
        maxx = max(list_fitness_raw)
        if maxx > (2 ** 64 - 1):
            list_fitness_normalized = self.generator.uniform(0, 0.1, self.pop_size)
        else:
            ### Normalize inflation rates (NI in Eq. (3.1) in the paper)
            list_fitness_normalized = np.reshape(self.normalize__(np.array([list_fitness_raw])), self.pop_size)  # Matrix
        pop_new = []
        for idx in range(0, self.pop_size):
            black_hole_pos = self.pop[idx].solution.copy()
            for jdx in range(0, self.problem.n_dims):
                r1 = self.generator.uniform()
                if r1 < list_fitness_normalized[idx]:
                    white_hole_id = self.roulette_wheel_selection__((-1. * list_fitness_raw))
                    if white_hole_id == None or white_hole_id == -1:
                        white_hole_id = 0
                    # Eq. (3.1) in the paper
                    black_hole_pos[jdx] = self.pop[white_hole_id].solution[jdx]
                # Eq. (3.2) in the paper if the boundaries are all the same
                r2 = self.generator.uniform()
                if r2 < wep:
                    r3 = self.generator.uniform()
                    if r3 < 0.5:
                        black_hole_pos[jdx] = self.g_best.solution[jdx] + tdr * self.generator.uniform(self.problem.lb[jdx], self.problem.ub[jdx])
                    else:
                        black_hole_pos[jdx] = self.g_best.solution[jdx] - tdr * self.generator.uniform(self.problem.lb[jdx], self.problem.ub[jdx])
            pos_new = self.correct_solution(black_hole_pos)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
