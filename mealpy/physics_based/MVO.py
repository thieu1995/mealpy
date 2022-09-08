# !/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseMVO(Optimizer):
    """
    The developed version: Multi-Verse Optimizer (MVO)

    Notes
    ~~~~~
    + New routtele wheel selection can handle negative values
    + Removed condition when np.random.normalize fitness. So the chance to choose while whole higher --> better

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wep_min (float): [0.05, 0.3], Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
        + wep_max (float: [0.75, 1.0], Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.MVO import BaseMVO
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
    >>> wep_min = 0.2
    >>> wep_max = 1.0
    >>> model = BaseMVO(epoch, pop_size, wep_min, wep_max)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, wep_min=0.2, wep_max=1.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wep_min (float): Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
            wep_max (float: Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.wep_min = self.validator.check_float("wep_min", wep_min, (0, 0.5))
        self.wep_max = self.validator.check_float("wep_max", wep_max, [0.5, 3.0])
        self.set_parameters(["epoch", "pop_size", "wep_min", "wep_max"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Eq. (3.3) in the paper
        wep = self.wep_max - (epoch + 1) * ((self.wep_max - self.wep_min) / self.epoch)

        # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        tdr = 1 - (epoch + 1) ** (1.0 / 6) / self.epoch ** (1.0 / 6)

        pop_new = []
        for idx in range(0, self.pop_size):
            if np.random.uniform() < wep:
                list_fitness = np.array([item[self.ID_TAR][self.ID_FIT] for item in self.pop])
                white_hole_id = self.get_index_roulette_wheel_selection(list_fitness)
                black_hole_pos_1 = self.pop[idx][self.ID_POS] + tdr * np.random.normal(0, 1) * \
                                   (self.pop[white_hole_id][self.ID_POS] - self.pop[idx][self.ID_POS])
                black_hole_pos_2 = self.g_best[self.ID_POS] + tdr * np.random.normal(0, 1) * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                black_hole_pos = np.where(np.random.random(self.problem.n_dims) < 0.5, black_hole_pos_1, black_hole_pos_2)
            else:
                black_hole_pos = self.generate_position(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position(black_hole_pos, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class OriginalMVO(BaseMVO):
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
    >>> from mealpy.physics_based.MVO import OriginalMVO
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
    >>> wep_min = 0.2
    >>> wep_max = 1.0
    >>> model = OriginalMVO(epoch, pop_size, wep_min, wep_max)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Mirjalili, S.M. and Hatamlou, A., 2016. Multi-verse optimizer: a nature-inspired
    algorithm for global optimization. Neural Computing and Applications, 27(2), pp.495-513.
    """

    def __init__(self, epoch=10000, pop_size=100, wep_min=0.2, wep_max=1.0, **kwargs):
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
        p = np.random.uniform() * accumulation[-1]
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
                return np.random.uniform(0.2, 0.8, self.pop_size)
            return d / np.sum(d, axis=0)
        else:
            ptp_vector = np.ptp(d, axis=0)
            if 0 in ptp_vector:
                return np.random.uniform(0.2, 0.8, self.pop_size)
            return d / np.ptp(d, axis=0)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Eq. (3.3) in the paper
        wep = self.wep_min + (epoch + 1) * ((self.wep_max - self.wep_min) / self.epoch)

        # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        tdr = 1 - (epoch + 1) ** (1.0 / 6) / self.epoch ** (1.0 / 6)

        list_fitness_raw = np.array([item[self.ID_TAR][self.ID_FIT] for item in self.pop])
        maxx = max(list_fitness_raw)
        if maxx > (2 ** 64 - 1):
            list_fitness_normalized = np.random.uniform(0, 0.1, self.pop_size)
        else:
            ### Normalize inflation rates (NI in Eq. (3.1) in the paper)
            list_fitness_normalized = np.reshape(self.normalize__(np.array([list_fitness_raw])), self.pop_size)  # Matrix

        pop_new = []
        for idx in range(0, self.pop_size):
            black_hole_pos = deepcopy(self.pop[idx][self.ID_POS])
            for j in range(0, self.problem.n_dims):
                r1 = np.random.uniform()
                if r1 < list_fitness_normalized[idx]:
                    white_hole_id = self.roulette_wheel_selection__((-1 * list_fitness_raw))
                    if white_hole_id == None or white_hole_id == -1:
                        white_hole_id = 0
                    # Eq. (3.1) in the paper
                    black_hole_pos[j] = self.pop[white_hole_id][self.ID_POS][j]

                # Eq. (3.2) in the paper if the boundaries are all the same
                r2 = np.random.uniform()
                if r2 < wep:
                    r3 = np.random.uniform()
                    if r3 < 0.5:
                        black_hole_pos[j] = self.g_best[self.ID_POS][j] + tdr * np.random.uniform(self.problem.lb[j], self.problem.ub[j])
                    else:
                        black_hole_pos[j] = self.g_best[self.ID_POS][j] - tdr * np.random.uniform(self.problem.lb[j], self.problem.ub[j])
            pos_new = self.amend_position(black_hole_pos, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
