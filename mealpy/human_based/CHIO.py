#!/usr/bin/env python
# Created by "Thieu" at 19:24, 09/05/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.target import Target


class OriginalCHIO(Optimizer):
    """
    The original version of: Coronavirus Herd Immunity Optimization (CHIO)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-020-05296-6

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + brr (float): [0.05, 0.2], Basic reproduction rate, default=0.15
        + max_age (int): [5, 20], Maximum infected cases age, default=10

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, CHIO
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
    >>> model = CHIO.OriginalCHIO(epoch=1000, pop_size=50, brr = 0.15, max_age = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Al-Betar, M.A., Alyasseri, Z.A.A., Awadallah, M.A. et al. Coronavirus herd immunity optimizer (CHIO).
    Neural Comput & Applic 33, 5011–5042 (2021). https://doi.org/10.1007/s00521-020-05296-6
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, brr: float = 0.15, max_age: int = 10, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            brr (float): Basic reproduction rate, default=0.15
            max_age (int): Maximum infected cases age, default=10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.brr = self.validator.check_float("brr", brr, (0, 1.0))
        self.max_age = self.validator.check_int("max_age", max_age, [1, 1+int(epoch/5)])
        self.set_parameters(["epoch", "pop_size", "brr", "max_age"])

    def initialize_variables(self):
        self.immunity_type_list = self.generator.integers(0, 3, self.pop_size)  # Randint [0, 1, 2]
        self.age_list = np.zeros(self.pop_size)  # Control the age of each position
        self.finished = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        is_corona_list = [False, ] * self.pop_size
        for i in range(0, self.pop_size):
            pos_new = self.pop[i].solution.copy()
            for j in range(0, self.problem.n_dims):
                rand = self.generator.uniform()
                if rand < (1.0 / 3) * self.brr:
                    idx_candidates = np.where(self.immunity_type_list == 1)  # Infected list
                    if idx_candidates[0].size == 0:
                        self.finished = True
                        # print("Epoch: {}, i: {}, immunity_list: {}".format(epoch, i, self.immunity_type_list))
                        break
                    idx_selected = self.generator.choice(idx_candidates[0])
                    pos_new[j] = self.pop[i].solution[j] + self.generator.uniform() * (self.pop[i].solution[j] - self.pop[idx_selected].solution[j])
                    is_corona_list[i] = True
                elif (1.0 / 3) * self.brr <= rand < (2.0 / 3) * self.brr:
                    idx_candidates = np.where(self.immunity_type_list == 0)  # Susceptible list
                    idx_selected = self.generator.choice(idx_candidates[0])
                    pos_new[j] = self.pop[i].solution[j] + self.generator.uniform() * (self.pop[i].solution[j] - self.pop[idx_selected].solution[j])
                elif (2.0 / 3) * self.brr <= rand < self.brr:
                    idx_candidates = np.where(self.immunity_type_list == 2)  # Immunity list
                    fit_list = np.array([self.pop[item].target.fitness for item in idx_candidates[0]])
                    idx_selected = idx_candidates[0][np.argmin(fit_list)]  # Found the index of best fitness
                    pos_new[j] = self.pop[i].solution[j] + self.generator.uniform() * (self.pop[i].solution[j] - self.pop[idx_selected].solution[j])
            if self.finished:
                break
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        if len(pop_new) != self.pop_size:
            pop_child = self.generate_population(self.pop_size - len(pop_new))
            pop_new = pop_new + pop_child
        for idx in range(0, self.pop_size):
            # Step 4: Update herd immunity population
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = pop_new[idx].copy()
            else:
                self.age_list[idx] += 1
            ## Calculate immunity mean of population
            fit_list = np.array([agent.target.fitness for agent in self.pop])
            delta_fx = np.mean(fit_list)
            if self.compare_fitness(pop_new[idx].target.fitness, delta_fx, self.problem.minmax) and self.immunity_type_list[idx] == 0 and is_corona_list[idx]:
                self.immunity_type_list[idx] = 1
                self.age_list[idx] = 1
            if self.compare_fitness(delta_fx, pop_new[idx].target.fitness, self.problem.minmax) and (self.immunity_type_list[idx] == 1):
                self.immunity_type_list[idx] = 2
                self.age_list[idx] = 0
            # Step 5: Fatality condition
            if (self.age_list[idx] >= self.max_age) and (self.immunity_type_list[idx] == 1):
                self.pop[idx] = self.generate_agent()
                self.immunity_type_list[idx] = 0
                self.age_list[idx] = 0


class DevCHIO(OriginalCHIO):
    """
    The developed version of: Coronavirus Herd Immunity Optimization (CHIO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + brr (float): [0.05, 0.2], Basic reproduction rate, default=0.15
        + max_age (int): [5, 20], Maximum infected cases age, default=10

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, CHIO
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
    >>> model = CHIO.DevCHIO(epoch=1000, pop_size=50, brr = 0.15, max_age = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, brr: float = 0.15, max_age: int = 10, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            brr (float): Basic reproduction rate, default=0.15
            max_age (int): Maximum infected cases age, default=10
        """
        super().__init__(epoch, pop_size, brr, max_age, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        is_corona_list = [False, ] * self.pop_size
        for i in range(0, self.pop_size):
            pos_new = self.pop[i].solution.copy()
            for j in range(0, self.problem.n_dims):
                rand = self.generator.uniform()
                if rand < (1.0 / 3) * self.brr:
                    idx_candidates = np.where(self.immunity_type_list == 1)  # Infected list
                    if idx_candidates[0].size == 0:
                        rand_choice = self.generator.choice(range(0, self.pop_size), int(0.33 * self.pop_size), replace=False)
                        self.immunity_type_list[rand_choice] = 1
                        idx_candidates = np.where(self.immunity_type_list == 1)
                    idx_selected = self.generator.choice(idx_candidates[0])
                    pos_new[j] = self.pop[i].solution[j] + self.generator.uniform() * (self.pop[i].solution[j] - self.pop[idx_selected].solution[j])
                    is_corona_list[i] = True
                elif (1.0 / 3) * self.brr <= rand < (2.0 / 3) * self.brr:
                    idx_candidates = np.where(self.immunity_type_list == 0)  # Susceptible list
                    if idx_candidates[0].size == 0:
                        rand_choice = self.generator.choice(range(0, self.pop_size), int(0.33 * self.pop_size), replace=False)
                        self.immunity_type_list[rand_choice] = 0
                        idx_candidates = np.where(self.immunity_type_list == 0)
                    idx_selected = self.generator.choice(idx_candidates[0])
                    pos_new[j] = self.pop[i].solution[j] + self.generator.uniform() * (self.pop[i].solution[j] - self.pop[idx_selected].solution[j])
                elif (2.0 / 3) * self.brr <= rand < self.brr:
                    idx_candidates = np.where(self.immunity_type_list == 2)  # Immunity list
                    fit_list = np.array([self.pop[item].target.fitness for item in idx_candidates[0]])
                    idx_selected = idx_candidates[0][np.argmin(fit_list)]  # Found the index of best fitness
                    pos_new[j] = self.pop[i].solution[j] + self.generator.uniform() * (self.pop[i].solution[j] - self.pop[idx_selected].solution[j])
            if self.finished:
                break
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)

        for idx in range(0, self.pop_size):
            # Step 4: Update herd immunity population
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = pop_new[idx].copy()
            else:
                self.age_list[idx] += 1
            ## Calculate immunity mean of population
            fit_list = np.array([agent.target.fitness for agent in self.pop])
            delta_fx = np.mean(fit_list)
            if self.compare_fitness(pop_new[idx].target.fitness, delta_fx, self.problem.minmax) and (self.immunity_type_list[idx] == 0) and is_corona_list[idx]:
                self.immunity_type_list[idx] = 1
                self.age_list[idx] = 1
            if self.compare_fitness(delta_fx, pop_new[idx].target.fitness, self.problem.minmax) and (self.immunity_type_list[idx] == 1):
                self.immunity_type_list[idx] = 2
                self.age_list[idx] = 0
            # Step 5: Fatality condition
            if (self.age_list[idx] >= self.max_age) and (self.immunity_type_list[idx] == 1):
                self.pop[idx] = self.generate_agent()
                self.immunity_type_list[idx] = 0
                self.age_list[idx] = 0
