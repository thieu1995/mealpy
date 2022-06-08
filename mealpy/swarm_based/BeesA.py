# !/usr/bin/env python
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from copy import deepcopy


class BaseBeesA(Optimizer):
    """
    The original version of: Bees Algorithm (BeesA)

    Links:
        1. https://www.sciencedirect.com/science/article/pii/B978008045157250081X
        2. https://www.tandfonline.com/doi/full/10.1080/23311916.2015.1091540

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + site_ratio (list, tuple): (selected_site_ratio, elite_site_ratio), default = (0.5, 0.4)
        + site_bee_ratio (list, tuple): (selected_site_bee_ratio, elite_site_bee_ratio), default = (0.1, 2)
        + dance_factor (list, tuple): (radius, reduction), default = (0.1, 0.99)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.BeesA import BaseBeesA
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
    >>> site_ratio = [0.5, 0.4]
    >>> site_bee_ratio = [0.1, 2]
    >>> dance_factor = [0.1, 0.99]
    >>> dance_radius_damp = 0.99
    >>> model = BaseBeesA(problem_dict1, epoch, pop_size, site_ratio, site_bee_ratio, dance_factor)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Pham, D.T., Ghanbarzadeh, A., Koç, E., Otri, S., Rahim, S. and Zaidi, M., 2006.
    The bees algorithm—a novel tool for complex optimisation problems. In Intelligent
    production machines and systems (pp. 454-459). Elsevier Science Ltd.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, site_ratio=(0.5, 0.4),
                 site_bee_ratio=(0.1, 2.0), dance_factor=(0.1, 0.99), **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            site_ratio (list, tuple): (selected_site_ratio, elite_site_ratio)
            site_bee_ratio (list, tuple): (selected_site_bee_ratio, elite_site_bee_ratio)
            dance_factor (list, tuple): Bees Dance Radius, Bees Dance Radius Reduction Rate
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        # (Scout Bee Count or Population Size, Selected Sites Count)
        self.site_ratio = self.validator.check_tuple_float("site_ratio (selected_site_ratio, elite_site_ratio)", site_ratio, ((0, 1.0), (0, 1.0)))
        # Scout Bee Count, Selected Sites Bee Count
        self.site_bee_ratio = self.validator.check_tuple_float("site_bee_ratio (selected_site_bee_ratio, elite_site_bee_ratio)", site_bee_ratio, ((0, 1.0), (0, 3.0)))
        self.dance_factor = self.validator.check_tuple_float("dance_factor (radius, reduction)", dance_factor, ((0, 1.0), (0, 1.0)))
        self.dance_radius_damp = self.dance_factor[1]

        # Initial Value of Dance Radius
        self.dance_radius = self.dance_factor[0]
        self.dyn_radius = self.dance_factor[0]
        self.n_selected_bees = int(round(self.site_ratio[0] * self.pop_size))
        self.n_elite_bees = int(round(self.site_ratio[1] * self.n_selected_bees))
        self.n_selected_bees_local = int(round(self.site_bee_ratio[0] * self.pop_size))
        self.n_elite_bees_local = int(round(self.site_bee_ratio[1] * self.n_selected_bees_local))
        self.nfe_per_epoch = self.n_elite_bees * self.n_elite_bees_local + self.pop_size - self.n_selected_bees + \
                             (self.n_selected_bees - self.n_elite_bees) * self.n_selected_bees_local
        self.sort_flag = True

    def perform_dance__(self, position, r):
        j = np.random.choice(range(0, self.problem.n_dims))
        position[j] = position[j] + r * np.random.uniform(-1, 1)
        return self.amend_position(position, self.problem.lb, self.problem.ub)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        pop_new = deepcopy(self.pop)
        for idx in range(0, self.pop_size):
            # Elite Sites
            if idx < self.n_elite_bees:
                nfe_epoch += self.n_elite_bees_local
                pop_child = []
                for j in range(0, self.n_elite_bees_local):
                    pos_new = self.perform_dance__(self.pop[idx][self.ID_POS], self.dyn_radius)
                    pop_child.append([pos_new, None])
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
                pop_child = self.update_target_wrapper_population(pop_child)
                _, local_best = self.get_global_best_solution(pop_child)
                if self.compare_agent(local_best, self.pop[idx]):
                    pop_new[idx] = local_best
            elif self.n_elite_bees <= idx < self.n_selected_bees:
                # Selected Non-Elite Sites
                nfe_epoch += self.n_selected_bees_local
                pop_child = []
                for j in range(0, self.n_selected_bees_local):
                    pos_new = self.perform_dance__(self.pop[idx][self.ID_POS], self.dyn_radius)
                    pop_child.append([pos_new, None])
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
                pop_child = self.update_target_wrapper_population(pop_child)
                _, local_best = self.get_global_best_solution(pop_child)
                if self.compare_agent(local_best, self.pop[idx]):
                    pop_new[idx] = local_best
            else:
                # Non-Selected Sites
                nfe_epoch += 1
                pop_new[idx] = self.create_solution(self.problem.lb, self.problem.ub)
        self.pop = pop_new
        # Damp Dance Radius
        self.dyn_radius = self.dance_radius_damp * self.dance_radius
        self.nfe_per_epoch = nfe_epoch


class ProbBeesA(Optimizer):
    """
    The original version of: Bees Algorithm (BeesA)

    Notes
    ~~~~~
    + This is probabilistic version of Bees Algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + recruited_bee_ratio (float): percent of bees recruited, default = 0.1
        + dance_factor (tuple, list): (radius, reduction) - Bees Dance Radius, default=(0.1, 0.99)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.BeesA import ProbBeesA
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
    >>> recruited_bee_ratio = 0.1
    >>> dance_factor = (0.1, 0.99)
    >>> model = ProbBeesA(problem_dict1, epoch, pop_size, recruited_bee_ratio, dance_factor)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Pham, D.T. and Castellani, M., 2015. A comparative study of the Bees Algorithm as a tool for
    function optimisation. Cogent Engineering, 2(1), p.1091540.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, recruited_bee_ratio=0.1, dance_factor=(0.1, 0.99), **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            recruited_bee_ratio (float): percent of bees recruited, default = 0.1
            dance_factor (tuple, list): Bees Dance Radius, Bees Dance Radius Reduction Rate, default=(0.1, 0.99)
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.recruited_bee_ratio = self.validator.check_float("recruited_bee_ratio", recruited_bee_ratio, (0, 1.0))
        self.dance_factor = self.validator.check_tuple_float("dance_factor (radius, reduction)", dance_factor, ((0, 1.0), (0, 1.0)))
        self.dance_radius = self.dance_factor[0]
        self.dance_radius_damp = self.dance_factor[1]

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True
        # Initial Value of Dance Radius
        self.dyn_radius = self.dance_radius
        self.recruited_bee_count = int(round(self.recruited_bee_ratio * self.pop_size))

    def perform_dance__(self, position, r):
        j = np.random.choice(list(range(0, self.problem.n_dims)))
        position[j] = position[j] + r * np.random.uniform(-1, 1)
        return self.amend_position(position, self.problem.lb, self.problem.ub)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate Scores
        fit_list = np.array([solution[self.ID_TAR][self.ID_FIT] for solution in self.pop])
        fit_list = 1.0 / fit_list
        d_fit = fit_list / np.mean(fit_list)

        nfe_epoch = 0
        for idx in range(0, self.pop_size):
            # Determine Rejection Probability based on Score
            if d_fit[idx] < 0.9:
                reject_prob = 0.6
            elif 0.9 <= d_fit[idx] < 0.95:
                reject_prob = 0.2
            elif 0.95 <= d_fit[idx] < 1.15:
                reject_prob = 0.05
            else:
                reject_prob = 0

            # Check for Acceptance/Rejection
            if np.random.rand() >= reject_prob:  # Acceptance
                # Calculate New Bees Count
                bee_count = int(np.ceil(d_fit[idx] * self.recruited_bee_count))
                if bee_count < 2: bee_count = 2
                if bee_count > self.pop_size: bee_count = self.pop_size
                # Create New Bees(Solutions)
                pop_child = []
                nfe_epoch += bee_count
                for j in range(0, bee_count):
                    pos_new = self.perform_dance__(self.pop[idx][self.ID_POS], self.dyn_radius)
                    pop_child.append([pos_new, None])
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
                pop_child = self.update_target_wrapper_population(pop_child)
                _, local_best = self.get_global_best_solution(pop_child)
                if self.compare_agent(local_best, self.pop[idx]):
                    self.pop[idx] = local_best
            else:
                nfe_epoch += 1
                self.pop[idx] = self.create_solution(self.problem.lb, self.problem.ub)
        self.nfe_per_epoch = nfe_epoch
        # Damp Dance Radius
        self.dyn_radius = self.dance_radius_damp * self.dance_radius
