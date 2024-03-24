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
    >>> from mealpy import FloatVar, AEO
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
    >>> model = AEO.OriginalAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Wang, L. and Zhang, Z., 2020. Artificial ecosystem-based optimization: a novel
    nature-inspired meta-heuristic algorithm. Neural Computing and Applications, 32(13), pp.9383-9425.
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
        ## Production   - Update the worst agent
        # Eq. 2, 3, 1
        a = (1.0 - epoch / self.epoch) * self.generator.uniform()
        x1 = (1 - a) * self.pop[-1].solution + a * self.generator.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.correct_solution(x1)
        agent = self.generate_agent(pos_new)
        self.pop[-1] = agent
        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            rand = self.generator.random()
            # Eq. 4, 5, 6
            v1 = self.generator.normal(0, 1)
            v2 = self.generator.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor
            jdx = 1 if idx == 0 else self.generator.integers(0, idx)
            ### Herbivore
            if rand < 1.0 / 3:
                x_t1 = self.pop[idx].solution + c * (self.pop[idx].solution - self.pop[0].solution)  # Eq. 6
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
                x_t1 = self.pop[idx].solution + c * (self.pop[idx].solution - self.pop[jdx].solution)  # Eq. 7
            ### Omnivore
            else:
                r2 = self.generator.uniform()
                x_t1 = self.pop[idx].solution + c * (r2 * (self.pop[idx].solution - self.pop[0].solution)
                                                         + (1 - r2) * (self.pop[idx].solution - self.pop[jdx].solution))
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new, self.problem.minmax)
        ## find current best used in decomposition
        best = self.get_best_agent(self.pop, self.problem.minmax)
        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx in range(0, self.pop_size):
            r3 = self.generator.uniform()
            d = 3 * self.generator.normal(0, 1)
            e = r3 * self.generator.integers(1, 3) - 1
            h = 2 * r3 - 1
            x_t1 = best.solution + d * (e * best.solution - h * self.pop[idx].solution)
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)


class ImprovedAEO(OriginalAEO):
    """
    The original version of: Improved Artificial Ecosystem-based Optimization (ImprovedAEO)

    Links:
        1. https://doi.org/10.1016/j.ijhydene.2020.06.256

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AEO
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
    >>> model = AEO.ImprovedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rizk-Allah, R.M. and El-Fergany, A.A., 2021. Artificial ecosystem optimizer for parameters identification of
    proton exchange membrane fuel cells model. International Journal of Hydrogen Energy, 46(75), pp.37612-37627.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
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
        a = (1.0 - epoch / self.epoch) * self.generator.uniform()
        x1 = (1 - a) * self.pop[-1].solution + a * self.generator.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.correct_solution(x1)
        agent = self.generate_agent(pos_new)
        self.pop[-1] = agent
        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            rand = self.generator.random()
            # Eq. 4, 5, 6
            v1 = self.generator.normal(0, 1)
            v2 = self.generator.normal(0, 1)
            c = 0.5 * v1 / np.abs(v2)  # Consumption factor
            j = 1 if idx == 0 else self.generator.integers(0, idx)
            ### Herbivore
            if rand < 1.0 / 3:
                x_t1 = self.pop[idx].solution + c * (self.pop[idx].solution - self.pop[0].solution)  # Eq. 6
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
                x_t1 = self.pop[idx].solution + c * (self.pop[idx].solution - self.pop[j].solution)  # Eq. 7
            ### Omnivore
            else:
                r2 = self.generator.uniform()
                x_t1 = self.pop[idx].solution + c * (r2 * (self.pop[idx].solution - self.pop[0].solution) +
                                                     (1 - r2) * (self.pop[idx].solution - self.pop[j].solution))
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new, self.problem.minmax)
        ## find current best used in decomposition
        best = self.get_best_agent(self.pop, self.problem.minmax)
        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx in range(0, self.pop_size):
            r3 = self.generator.uniform()
            d = 3 * self.generator.normal(0, 1)
            e = r3 * self.generator.integers(1, 3) - 1
            h = 2 * r3 - 1
            if self.generator.random() < 0.5:
                beta = 1 - (1 - 0) * (epoch/ self.epoch)  # Eq. 21
                x_r = self.pop[self.generator.integers(0, self.pop_size - 1)].solution
                if self.generator.random() < 0.5:
                    x_new = beta * x_r + (1 - beta) * self.pop[idx].solution
                else:
                    x_new = beta * self.pop[idx].solution + (1 - beta) * x_r
            else:
                x_new = best.solution + d * (e * best.solution - h * self.pop[idx].solution)
                # x_new = best.solution + self.generator.normal() * best.solution
            pos_new = self.correct_solution(x_new)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)


class EnhancedAEO(Optimizer):
    """
    The original version of: Enhanced Artificial Ecosystem-Based Optimization (EAEO)

    Links:
        1. https://doi.org/10.1109/ACCESS.2020.3027654

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AEO
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
    >>> model = AEO.EnhancedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Eid, A., Kamel, S., Korashy, A. and Khurshaid, T., 2020. An enhanced artificial ecosystem-based
    optimization for optimal allocation of multiple distributed generations. IEEE Access, 8, pp.178493-178513.
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
        ## Production - Update the worst agent
        # Eq. 13
        a = 2 * (1. - epoch / self.epoch)
        x1 = (1 - a) * self.pop[-1].solution + a * self.generator.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.correct_solution(x1)
        agent = self.generate_agent(pos_new)
        self.pop[-1] = agent
        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            rand = self.generator.random()
            # Eq. 4, 5, 6
            v1 = self.generator.normal(0, 1)
            v2 = self.generator.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor
            r3 = 2 * np.pi * self.generator.random()
            r4 = self.generator.random()
            j = 1 if idx == 0 else self.generator.integers(0, idx)
            ### Herbivore
            if rand <= 1.0 / 3:  # Eq. 15
                if r4 <= 0.5:
                    x_t1 = self.pop[idx].solution + np.sin(r3) * c * (self.pop[idx].solution - self.pop[0].solution)
                else:
                    x_t1 = self.pop[idx].solution + np.cos(r3) * c * (self.pop[idx].solution - self.pop[0].solution)
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:  # Eq. 16
                if r4 <= 0.5:
                    x_t1 = self.pop[idx].solution + np.sin(r3) * c * (self.pop[idx].solution - self.pop[j].solution)
                else:
                    x_t1 = self.pop[idx].solution + np.cos(r3) * c * (self.pop[idx].solution - self.pop[j].solution)
            ### Omnivore
            else:  # Eq. 17
                r5 = self.generator.random()
                if r4 <= 0.5:
                    x_t1 = self.pop[idx].solution + np.sin(r5) * c * (r5 * (self.pop[idx].solution - self.pop[0].solution) +
                                                                          (1 - r5) * (self.pop[idx].solution - self.pop[j].solution))
                else:
                    x_t1 = self.pop[idx].solution + np.cos(r5) * c * (r5 * (self.pop[idx].solution - self.pop[0].solution) +
                                                                          (1 - r5) * (self.pop[idx].solution - self.pop[j].solution))
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new, self.problem.minmax)
        ## find current best used in decomposition
        best = self.get_best_agent(self.pop, self.problem.minmax)
        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx in range(0, self.pop_size):
            r3 = self.generator.uniform()
            d = 3 * self.generator.normal(0, 1)
            e = r3 * self.generator.integers(1, 3) - 1
            h = 2 * r3 - 1
            if self.generator.random() < 0.5:
                beta = 1 - (1 - 0) * (epoch / self.epoch)  # Eq. 21
                r_idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                x_r = self.pop[r_idx].solution
                if self.generator.random() < 0.5:
                    x_new = beta * x_r + (1 - beta) * self.pop[idx].solution
                else:
                    x_new = (1 - beta) * x_r + beta * self.pop[idx].solution
            else:
                x_new = best.solution + d * (e * best.solution - h * self.pop[idx].solution)
            pos_new = self.correct_solution(x_new)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)


class ModifiedAEO(Optimizer):
    """
    The original version of: Modified Artificial Ecosystem-Based Optimization (MAEO)

    Links:
        1. https://doi.org/10.1109/ACCESS.2020.2973351

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AEO
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
    >>> model = AEO.ModifiedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Menesy, A.S., Sultan, H.M., Korashy, A., Banakhr, F.A., Ashmawy, M.G. and Kamel, S., 2020. Effective
    parameter extraction of different polymer electrolyte membrane fuel cell stack models using a
    modified artificial ecosystem optimization algorithm. IEEE Access, 8, pp.31892-31909.
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
        ## Production
        # Eq. 22
        H = 2 * (1 - epoch / self.epoch)
        a = (1 - epoch / self.epoch) * self.generator.random()
        x1 = (1 - a) * self.pop[-1].solution + a * self.generator.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.correct_solution(x1)
        agent = self.generate_agent(pos_new)
        self.pop[-1] = agent
        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            rand = self.generator.random()
            # Eq. 4, 5, 6
            v1 = self.generator.normal(0, 1)
            v2 = self.generator.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor
            j = 1 if idx == 0 else self.generator.integers(0, idx)
            ### Herbivore
            if rand <= 1.0 / 3:  # Eq. 23
                pos_new = self.pop[idx].solution + H * c * (self.pop[idx].solution - self.pop[0].solution)
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:  # Eq. 24
                pos_new = self.pop[idx].solution + H * c * (self.pop[idx].solution - self.pop[j].solution)
            ### Omnivore
            else:  # Eq. 25
                r5 = self.generator.random()
                pos_new = self.pop[idx].solution + H * c * (r5 * (self.pop[idx].solution - self.pop[0].solution) +
                                                                (1 - r5) * (self.pop[idx].solution - self.pop[j].solution))
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new, self.problem.minmax)
        ## find current best used in decomposition
        best = self.get_best_agent(self.pop, self.problem.minmax)
        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx in range(0, self.pop_size):
            r3 = self.generator.uniform()
            d = 3 * self.generator.normal(0, 1)
            e = r3 * self.generator.integers(1, 3) - 1
            h = 2 * r3 - 1
            if self.generator.random() < 0.5:
                beta = 1 - (1 - 0) * (epoch / self.epoch)  # Eq. 21
                r_idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                x_r = self.pop[r_idx].solution
                if self.generator.random() < 0.5:
                    x_new = beta * x_r + (1 - beta) * self.pop[idx].solution
                else:
                    x_new = (1 - beta) * x_r + beta * self.pop[idx].solution
            else:
                x_new = best.solution + d * (e * best.solution - h * self.pop[idx].solution)
            pos_new = self.correct_solution(x_new)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)


class AugmentedAEO(Optimizer):
    """
    The original version of: Augmented Artificial Ecosystem Optimization (AAEO)

    Notes:
        + Used linear weight factor reduce from 2 to 0 through time
        + Applied Levy-flight technique and the global best solution

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AEO
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
    >>> model = AEO.AugmentedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Van Thieu, N., Barma, S. D., Van Lam, T., Kisi, O., & Mahesha, A. (2022). Groundwater level modeling
    using Augmented Artificial Ecosystem Optimization. Journal of Hydrology, 129034.
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
        ## Production - Update the worst agent
        # Eq. 2, 3, 1
        wf = 2 * (1 - epoch / self.epoch)  # Weight factor
        a = (1.0 - epoch / self.epoch) * self.generator.random()
        x1 = (1 - a) * self.pop[-1].solution + a * self.generator.uniform(self.problem.lb, self.problem.ub)
        pos_new = self.correct_solution(x1)
        agent = self.generate_agent(pos_new)
        self.pop[-1] = agent
        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            if self.generator.random() < 0.5:
                rand = self.generator.random()
                # Eq. 4, 5, 6
                c = 0.5 * self.generator.normal(0, 1) / np.abs(self.generator.normal(0, 1))  # Consumption factor
                j = 1 if idx == 0 else self.generator.integers(0, idx)
                ### Herbivore
                if rand < 1.0 / 3:
                    pos_new = self.pop[idx].solution + wf * c * (self.pop[idx].solution - self.pop[0].solution)  # Eq. 6
                ### Omnivore
                elif 1.0 / 3 <= rand <= 2.0 / 3:
                    pos_new = self.pop[idx].solution + wf * c * (self.pop[idx].solution - self.pop[j].solution)  # Eq. 7
                ### Carnivore
                else:
                    r2 = self.generator.uniform()
                    pos_new = self.pop[idx].solution + wf * c * (r2 * (self.pop[idx].solution - self.pop[0].solution) +
                                                                     (1 - r2) * (self.pop[idx].solution - self.pop[j].solution))
            else:
                pos_new = self.pop[idx].solution + self.get_levy_flight_step(1., 0.001, case=-1) * \
                          (1.0 / np.sqrt(epoch)) * np.sign(self.generator.random() - 0.5) * (self.pop[idx].solution - self.g_best.solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new, self.problem.minmax)
        ## find current best used in decomposition
        best = self.get_best_agent(self.pop, self.problem.minmax)
        ## Decomposition
        ### Eq. 10, 11, 12, 9   idx, pop, g_best, local_best
        pop_child = []
        for idx in range(0, self.pop_size):
            if self.generator.random() < 0.5:
                pos_new = best.solution + self.generator.normal(0, 1, self.problem.n_dims) * (best.solution - self.pop[idx].solution)
            else:
                beta = self.generator.uniform(0.01, 1.)
                pos_new = best.solution + self.get_levy_flight_step(beta=beta, multiplier=0.01, size=self.problem.n_dims, case=0) * (best.solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)
