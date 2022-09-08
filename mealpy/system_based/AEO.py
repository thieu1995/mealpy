#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAEO(Optimizer):
    """
    The original version of: Artificial Ecosystem-based Optimization (AEO)

    Links:
        1. https://doi.org/10.1007/s00521-019-04452-x
        2. https://www.mathworks.com/matlabcentral/fileexchange/72685-artificial-ecosystem-based-optimization-aeo

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.system_based.AEO import OriginalAEO
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
    >>> model = OriginalAEO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Wang, L. and Zhang, Z., 2020. Artificial ecosystem-based optimization: a novel
    nature-inspired meta-heuristic algorithm. Neural Computing and Applications, 32(13), pp.9383-9425.
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
        
        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Production   - Update the worst agent
        # Eq. 2, 3, 1
        a = (1.0 - epoch / self.epoch) * np.random.uniform()
        x1 = (1 - a) * self.pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.amend_position(x1, self.problem.lb, self.problem.ub)
        target = self.get_target_wrapper(pos_new)
        self.pop[-1] = [pos_new, target]

        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            rand = np.random.random()
            # Eq. 4, 5, 6
            v1 = np.random.normal(0, 1)
            v2 = np.random.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor
            j = 1 if idx == 0 else np.random.randint(0, idx)
            ### Herbivore
            if rand < 1.0 / 3:
                x_t1 = self.pop[idx][self.ID_POS] + c * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS])  # Eq. 6
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
                x_t1 = self.pop[idx][self.ID_POS] + c * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS])  # Eq. 7
            ### Omnivore
            else:
                r2 = np.random.uniform()
                x_t1 = self.pop[idx][self.ID_POS] + c * (r2 * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS])
                                                         + (1 - r2) * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS]))
            pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new)

        ## find current best used in decomposition
        _, best = self.get_global_best_solution(self.pop)

        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx in range(0, self.pop_size):
            r3 = np.random.uniform()
            d = 3 * np.random.normal(0, 1)
            e = r3 * np.random.randint(1, 3) - 1
            h = 2 * r3 - 1
            x_t1 = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)


class ImprovedAEO(OriginalAEO):
    """
    The original version of: Improved Artificial Ecosystem-based Optimization (ImprovedAEO)

    Links:
        1. https://doi.org/10.1016/j.ijhydene.2020.06.256

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.system_based.AEO import ImprovedAEO
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
    >>> model = ImprovedAEO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Rizk-Allah, R.M. and El-Fergany, A.A., 2021. Artificial ecosystem optimizer
    for parameters identification of proton exchange membrane fuel cells model.
    International Journal of Hydrogen Energy, 46(75), pp.37612-37627.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Production   - Update the worst agent
        # Eq. 2, 3, 1
        a = (1.0 - epoch / self.epoch) * np.random.uniform()
        x1 = (1 - a) * self.pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.amend_position(x1, self.problem.lb, self.problem.ub)
        target = self.get_target_wrapper(pos_new)
        self.pop[-1] = [pos_new, target]

        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            rand = np.random.random()
            # Eq. 4, 5, 6
            v1 = np.random.normal(0, 1)
            v2 = np.random.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor
            j = 1 if idx == 0 else np.random.randint(0, idx)
            ### Herbivore
            if rand < 1.0 / 3:
                x_t1 = self.pop[idx][self.ID_POS] + c * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS])  # Eq. 6
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
                x_t1 = self.pop[idx][self.ID_POS] + c * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS])  # Eq. 7
            ### Omnivore
            else:
                r2 = np.random.uniform()
                x_t1 = self.pop[idx][self.ID_POS] + c * (r2 * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS])
                                                         + (1 - r2) * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS]))
            pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new)

        ## find current best used in decomposition
        _, best = self.get_global_best_solution(self.pop)

        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx in range(0, self.pop_size):
            r3 = np.random.uniform()
            d = 3 * np.random.normal(0, 1)
            e = r3 * np.random.randint(1, 3) - 1
            h = 2 * r3 - 1

            x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * self.pop[idx][self.ID_POS])
            if np.random.random() < 0.5:
                beta = 1 - (1 - 0) * ((epoch + 1) / self.epoch)  # Eq. 21
                x_r = self.pop[np.random.randint(0, self.pop_size - 1)][self.ID_POS]
                if np.random.random() < 0.5:
                    x_new = beta * x_r + (1 - beta) * self.pop[idx][self.ID_POS]
                else:
                    x_new = beta * self.pop[idx][self.ID_POS] + (1 - beta) * x_r
            else:
                best[self.ID_POS] = best[self.ID_POS] + np.random.normal() * best[self.ID_POS]
            pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)


class EnhancedAEO(Optimizer):
    """
    The original version of: Enhanced Artificial Ecosystem-Based Optimization (EAEO)

    Links:
        1. https://doi.org/10.1109/ACCESS.2020.3027654

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.system_based.AEO import EnhancedAEO
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
    >>> model = EnhancedAEO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Eid, A., Kamel, S., Korashy, A. and Khurshaid, T., 2020. An enhanced artificial ecosystem-based
    optimization for optimal allocation of multiple distributed generations. IEEE Access, 8, pp.178493-178513.
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
        
        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Production - Update the worst agent
        # Eq. 13
        a = 2 * (1 - (epoch + 1) / self.epoch)
        x1 = (1 - a) * self.pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.amend_position(x1, self.problem.lb, self.problem.ub)
        target = self.get_target_wrapper(pos_new)
        self.pop[-1] = [pos_new, target]

        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            rand = np.random.random()
            # Eq. 4, 5, 6
            v1 = np.random.normal(0, 1)
            v2 = np.random.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor

            r3 = 2 * np.pi * np.random.random()
            r4 = np.random.random()
            j = 1 if idx == 0 else np.random.randint(0, idx)
            ### Herbivore
            if rand <= 1.0 / 3:  # Eq. 15
                if r4 <= 0.5:
                    x_t1 = self.pop[idx][self.ID_POS] + np.sin(r3) * c * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS])
                else:
                    x_t1 = self.pop[idx][self.ID_POS] + np.cos(r3) * c * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS])
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:  # Eq. 16
                if r4 <= 0.5:
                    x_t1 = self.pop[idx][self.ID_POS] + np.sin(r3) * c * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS])
                else:
                    x_t1 = self.pop[idx][self.ID_POS] + np.cos(r3) * c * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS])
            ### Omnivore
            else:  # Eq. 17
                r5 = np.random.random()
                if r4 <= 0.5:
                    x_t1 = self.pop[idx][self.ID_POS] + np.sin(r5) * c * (r5 * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS]) +
                                                                          (1 - r5) * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS]))
                else:
                    x_t1 = self.pop[idx][self.ID_POS] + np.cos(r5) * c * (r5 * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS]) +
                                                                          (1 - r5) * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS]))
            pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new)

        ## find current best used in decomposition
        _, best = self.get_global_best_solution(self.pop)

        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx in range(0, self.pop_size):
            r3 = np.random.uniform()
            d = 3 * np.random.normal(0, 1)
            e = r3 * np.random.randint(1, 3) - 1
            h = 2 * r3 - 1
            # x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * agent_i[self.ID_POS])
            if np.random.random() < 0.5:
                beta = 1 - (1 - 0) * ((epoch + 1) / self.epoch)  # Eq. 21
                r_idx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                x_r = self.pop[r_idx][self.ID_POS]
                # x_r = pop[np.random.randint(0, self.pop_size-1)][self.ID_POS]
                if np.random.random() < 0.5:
                    x_new = beta * x_r + (1 - beta) * self.pop[idx][self.ID_POS]
                else:
                    x_new = (1 - beta) * x_r + beta * self.pop[idx][self.ID_POS]
            else:
                x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * self.pop[idx][self.ID_POS])
                # x_new = best[self.ID_POS] + np.random.normal() * best[self.ID_POS]
            pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)


class ModifiedAEO(Optimizer):
    """
    The original version of: Modified Artificial Ecosystem-Based Optimization (MAEO)

    Links:
        1. https://doi.org/10.1109/ACCESS.2020.2973351

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.system_based.AEO import ModifiedAEO
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
    >>> model = ModifiedAEO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Menesy, A.S., Sultan, H.M., Korashy, A., Banakhr, F.A., Ashmawy, M.G. and Kamel, S., 2020. Effective
    parameter extraction of different polymer electrolyte membrane fuel cell stack models using a
    modified artificial ecosystem optimization algorithm. IEEE Access, 8, pp.31892-31909.
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
        
        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Production
        # Eq. 22
        H = 2 * (1 - (epoch + 1) / self.epoch)
        a = (1 - (epoch + 1) / self.epoch) * np.random.random()
        x1 = (1 - a) * self.pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.amend_position(x1, self.problem.lb, self.problem.ub)
        target = self.get_target_wrapper(pos_new)
        self.pop[-1] = [pos_new, target]

        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            rand = np.random.random()
            # Eq. 4, 5, 6
            v1 = np.random.normal(0, 1)
            v2 = np.random.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor
            j = 1 if idx == 0 else np.random.randint(0, idx)
            ### Herbivore
            if rand <= 1.0 / 3:  # Eq. 23
                pos_new = self.pop[idx][self.ID_POS] + H * c * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS])
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:  # Eq. 24
                pos_new = self.pop[idx][self.ID_POS] + H * c * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS])
            ### Omnivore
            else:  # Eq. 25
                r5 = np.random.random()
                pos_new = self.pop[idx][self.ID_POS] + H * c * (r5 * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS]) +
                                                                (1 - r5) * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS]))
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new)

        ## find current best used in decomposition
        _, best = self.get_global_best_solution(self.pop)

        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx in range(0, self.pop_size):
            r3 = np.random.uniform()
            d = 3 * np.random.normal(0, 1)
            e = r3 * np.random.randint(1, 3) - 1
            h = 2 * r3 - 1
            # x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * agent_i[self.ID_POS])
            if np.random.random() < 0.5:
                beta = 1 - (1 - 0) * ((epoch + 1) / self.epoch)  # Eq. 21
                r_idx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                x_r = self.pop[r_idx][self.ID_POS]
                # x_r = pop[np.random.randint(0, self.pop_size-1)][self.ID_POS]
                if np.random.random() < 0.5:
                    x_new = beta * x_r + (1 - beta) * self.pop[idx][self.ID_POS]
                else:
                    x_new = (1 - beta) * x_r + beta * self.pop[idx][self.ID_POS]
            else:
                x_new = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * self.pop[idx][self.ID_POS])
                # x_new = best[self.ID_POS] + np.random.normal() * best[self.ID_POS]
            pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)


class AdaptiveAEO(Optimizer):
    """
    The original version of: Adaptive Artificial Ecosystem Optimization (AAEO)

    Notes
    ~~~~~
    + Used linear weight factor reduce from 2 to 0 through time
    + Applied Levy-flight technique and the global best solution

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.system_based.AEO import AdaptiveAEO
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
    >>> model = AdaptiveAEO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Under Review
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
        
        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Production - Update the worst agent
        # Eq. 2, 3, 1
        wf = 2 * (1 - (epoch + 1) / self.epoch)  # Weight factor
        a = (1.0 - epoch / self.epoch) * np.random.random()
        x1 = (1 - a) * self.pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.amend_position(x1, self.problem.lb, self.problem.ub)
        target = self.get_target_wrapper(pos_new)
        self.pop[-1] = [pos_new, target]

        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            if np.random.random() < 0.5:
                rand = np.random.random()
                # Eq. 4, 5, 6
                c = 0.5 * np.random.normal(0, 1) / abs(np.random.normal(0, 1))  # Consumption factor
                j = 1 if idx == 0 else np.random.randint(0, idx)
                ### Herbivore
                if rand < 1.0 / 3:
                    pos_new = self.pop[idx][self.ID_POS] + wf * c * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS])  # Eq. 6
                ### Omnivore
                elif 1.0 / 3 <= rand <= 2.0 / 3:
                    pos_new = self.pop[idx][self.ID_POS] + wf * c * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS])  # Eq. 7
                ### Carnivore
                else:
                    r2 = np.random.uniform()
                    pos_new = self.pop[idx][self.ID_POS] + wf * c * (r2 * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS]) +
                                                                     (1 - r2) * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS]))
            else:
                pos_new = self.pop[idx][self.ID_POS] + self.get_levy_flight_step(1., 0.0001, case=-1) * \
                          (1.0 / np.sqrt(epoch + 1)) * np.sign(np.random.random() - 0.5) * (self.pop[idx][self.ID_POS] - self.g_best[self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new)

        ## find current best used in decomposition
        _, best = self.get_global_best_solution(self.pop)

        ## Decomposition
        ### Eq. 10, 11, 12, 9   idx, pop, g_best, local_best
        pop_child = []
        for idx in range(0, self.pop_size):
            if np.random.random() < 0.5:
                pos_new = best[self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * (best[self.ID_POS] - self.pop[idx][self.ID_POS])
            else:
                pos_new = best[self.ID_POS] + self.get_levy_flight_step(0.75, 0.001, case=-1) * \
                          1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * (best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)
