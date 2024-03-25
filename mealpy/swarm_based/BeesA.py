#!/usr/bin/env python
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class CleverBookBeesA(Optimizer):
    """
    The original version of: Bees Algorithm

    Notes:
        + This version is based on ABC in the book Clever Algorithms
        + Improved the function search_neighborhood__

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_elites (int): number of employed bees which provided for good location
        + n_others (int): number of employed bees which provided for other location
        + patch_size (float): patch_variables = patch_variables * patch_reduction
        + patch_reduction (float): the reduction factor
        + n_sites (int): 3 bees (employed bees, onlookers and scouts),
        + n_elite_sites (int): 1 good partition

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BeesA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BeesA.CleverBookBeesA(epoch=1000, pop_size=50, n_elites = 16, n_others = 4,
    >>>             patch_size = 5.0, patch_reduction = 0.985, n_sites = 3, n_elite_sites = 1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] D. T. Pham, Ghanbarzadeh A., Koc E., Otri S., Rahim S., and M.Zaidi. The bees algorithm - a novel tool
    for complex optimisation problems. In Proceedings of IPROMS 2006 Conference, pages 454–461, 2006.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_elites: int = 16, n_others: int = 4,
                 patch_size: float = 5.0, patch_reduction: float = 0.985, n_sites: int = 3, n_elite_sites: int = 1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_elites (int): number of employed bees which provided for good location
            n_others (int): number of employed bees which provided for other location
            patch_size (float): patch_variables = patch_variables * patch_reduction
            patch_reduction (float): the reduction factor
            n_sites (int): 3 bees (employed bees, onlookers and scouts),
            n_elite_sites (int): 1 good partition
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_elites = self.validator.check_int("n_elites", n_elites, [4, 20])
        self.n_others = self.validator.check_int("n_others", n_others, [2, 5])
        self.patch_size = self.validator.check_float("patch_size", patch_size, [2, 10])
        self.patch_reduction = self.validator.check_float("patch_reduction", patch_reduction, (0, 1.0))
        self.n_sites = self.validator.check_int("n_sites", n_sites, [2, 5])
        self.n_elite_sites = self.validator.check_int("n_elite_sites", n_elite_sites, [1, 3])
        self.set_parameters(["epoch", "pop_size", "n_elites", "n_others", "patch_size", "patch_reduction", "n_sites", "n_elite_sites"])
        self.sort_flag = True

    def search_neighborhood__(self, parent=None, neigh_size=None):
        """
        Search 1 best position in neigh_size position
        """
        pop_neigh = []
        for idx in range(0, neigh_size):
            t1 = self.generator.integers(0, len(parent.solution) - 1)
            new_bee = parent.solution.copy()
            new_bee[t1] = (parent.solution[t1] + self.generator.uniform() * self.patch_size) if self.generator.uniform() < 0.5 \
                else (parent.solution[t1] - self.generator.uniform() * self.patch_size)
            pos_new = self.correct_solution(new_bee)
            agent = self.generate_empty_agent(pos_new)
            pop_neigh.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_neigh[-1].target = self.get_target(pos_new)
        pop_neigh = self.update_target_for_population(pop_neigh)
        return self.get_best_agent(pop_neigh, self.problem.minmax)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx < self.n_sites:
                if idx < self.n_elite_sites:
                    neigh_size = self.n_elites
                else:
                    neigh_size = self.n_others
                agent = self.search_neighborhood__(self.pop[idx], neigh_size)
            else:
                agent = self.generate_agent()
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class OriginalBeesA(Optimizer):
    """
    The original version of: Bees Algorithm (BeesA)

    Links:
        1. https://www.sciencedirect.com/science/article/pii/B978008045157250081X
        2. https://www.tandfonline.com/doi/full/10.1080/23311916.2015.1091540

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + selected_site_ratio (float): default = 0.5
        + elite_site_ratio (float): default = 0.4
        + selected_site_bee_ratio (float): default = 0.1
        + elite_site_bee_ratio (float): default = 2.0
        + dance_radius (float): default = 0.1
        + dance_reduction (float): default = 0.99

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BeesA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BeesA.OriginalBeesA(epoch=1000, pop_size=50, selected_site_ratio=0.5, elite_site_ratio=0.4,
    >>>         selected_site_bee_ratio=0.1, elite_site_bee_ratio=2.0, dance_radius=0.1, dance_reduction=0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Pham, D.T., Ghanbarzadeh, A., Koç, E., Otri, S., Rahim, S. and Zaidi, M., 2006. The bees algorithm—a novel tool
    for complex optimisation problems. In Intelligent production machines and systems (pp. 454-459). Elsevier Science Ltd.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, selected_site_ratio: float = 0.5,
                 elite_site_ratio: float = 0.4, selected_site_bee_ratio: float = 0.1, elite_site_bee_ratio: float = 2.0,
                 dance_radius: float = 0.1, dance_reduction: float = 0.99, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            selected_site_ratio (float): 
            elite_site_ratio (float):
            selected_site_bee_ratio (float): 
            elite_site_bee_ratio (float): 
            dance_radius (float): 
            dance_reduction (float): 
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        # (Scout Bee Count or Population Size, Selected Sites Count)
        self.selected_site_ratio = self.validator.check_float("selected_site_ratio", selected_site_ratio, (0, 1.0))
        self.elite_site_ratio = self.validator.check_float("elite_site_ratio", elite_site_ratio, (0, 1.0))
        self.selected_site_bee_ratio = self.validator.check_float("selected_site_bee_ratio", selected_site_bee_ratio, (0, 1.0))
        self.elite_site_bee_ratio = self.validator.check_float("elite_site_bee_ratio", elite_site_bee_ratio, (0, 3.0))
        self.dance_radius = self.validator.check_float("dance_radius", dance_radius, (0, 1.0))
        self.dance_reduction = self.validator.check_float("dance_reduction", dance_reduction, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "selected_site_ratio", "elite_site_ratio", "selected_site_bee_ratio", 
                             "elite_site_bee_ratio", "dance_radius", "dance_reduction"])
        # Initial Value of Dance Radius
        self.dyn_radius = self.dance_radius
        self.n_selected_bees = int(round(self.selected_site_ratio * self.pop_size))
        self.n_elite_bees = int(round(self.elite_site_ratio * self.n_selected_bees))
        self.n_selected_bees_local = int(round(self.selected_site_bee_ratio * self.pop_size))
        self.n_elite_bees_local = int(round(self.elite_site_bee_ratio * self.n_selected_bees_local))
        self.sort_flag = True

    def perform_dance__(self, position, r):
        jdx = self.generator.choice(range(0, self.problem.n_dims))
        position[jdx] = position[jdx] + r * self.generator.uniform(-1, 1)
        return self.correct_solution(position)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = self.pop.copy()
        for idx in range(0, self.pop_size):
            # Elite Sites
            if idx < self.n_elite_bees:
                pop_child = []
                for j in range(0, self.n_elite_bees_local):
                    pos_new = self.perform_dance__(self.pop[idx].solution, self.dyn_radius)
                    agent = self.generate_empty_agent(pos_new)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].target = self.get_target(pos_new)
                pop_child = self.update_target_for_population(pop_child)
                local_best = self.get_best_agent(pop_child, self.problem.minmax)
                if self.compare_target(local_best.target, self.pop[idx].target, self.problem.minmax):
                    pop_new[idx] = local_best
            elif self.n_elite_bees <= idx < self.n_selected_bees:
                # Selected Non-Elite Sites
                pop_child = []
                for j in range(0, self.n_selected_bees_local):
                    pos_new = self.perform_dance__(self.pop[idx].solution, self.dyn_radius)
                    agent = self.generate_empty_agent(pos_new)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].target = self.get_target(pos_new)
                pop_child = self.update_target_for_population(pop_child)
                local_best = self.get_best_agent(pop_child, self.problem.minmax)
                if self.compare_target(local_best.target, self.pop[idx].target, self.problem.minmax):
                    pop_new[idx] = local_best
            else:
                # Non-Selected Sites
                pop_new[idx] = self.generate_agent()
        self.pop = pop_new
        # Damp Dance Radius
        self.dyn_radius = self.dance_reduction * self.dance_radius


class ProbBeesA(Optimizer):
    """
    The original version of: Probabilistic Bees Algorithm (BeesA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + recruited_bee_ratio (float): percent of bees recruited, default = 0.1
        + dance_factor (tuple, list): (radius, reduction) - Bees Dance Radius, default=(0.1, 0.99)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BeesA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BeesA.ProbBeesA(epoch=1000, pop_size=50, recruited_bee_ratio = 0.1, dance_radius = 0.1, dance_reduction = 0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Pham, D.T. and Castellani, M., 2015. A comparative study of the Bees Algorithm as a tool for
    function optimisation. Cogent Engineering, 2(1), p.1091540.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, recruited_bee_ratio: float = 0.1,
                 dance_radius: float = 0.1, dance_reduction: float = 0.99, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            recruited_bee_ratio (float): percent of bees recruited, default = 0.1
            dance_radius (float): Bees Dance Radius, default=0.1
            dance_reduction (float): Bees Dance Radius Reduction Rate, default=0.99
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.recruited_bee_ratio = self.validator.check_float("recruited_bee_ratio", recruited_bee_ratio, (0, 1.0))
        self.dance_radius = self.validator.check_float("dance_radius", dance_radius, (0, 1.0))
        self.dance_reduction = self.validator.check_float("dance_reduction", dance_reduction, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "recruited_bee_ratio", "dance_radius", "dance_reduction"])
        self.sort_flag = True
        # Initial Value of Dance Radius
        self.dyn_radius = self.dance_radius
        self.recruited_bee_count = int(round(self.recruited_bee_ratio * self.pop_size))

    def perform_dance__(self, position, r):
        jdx = self.generator.choice(list(range(0, self.problem.n_dims)))
        position[jdx] = position[jdx] + r * self.generator.uniform(-1, 1)
        return self.correct_solution(position)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate Scores
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        fit_list = 1.0 / fit_list
        d_fit = fit_list / np.mean(fit_list)
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
            if self.generator.random() >= reject_prob:  # Acceptance
                # Calculate New Bees Count
                bee_count = int(np.ceil(d_fit[idx] * self.recruited_bee_count))
                if bee_count < 2: bee_count = 2
                if bee_count > self.pop_size: bee_count = self.pop_size
                # Create New Bees(Solutions)
                pop_child = []
                for j in range(0, bee_count):
                    pos_new = self.perform_dance__(self.pop[idx].solution, self.dyn_radius)
                    agent = self.generate_empty_agent(pos_new)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].target = self.get_target(pos_new)
                pop_child = self.update_target_for_population(pop_child)
                local_best = self.get_best_agent(pop_child, self.problem.minmax)
                if self.compare_target(local_best.target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = local_best
            else:
                self.pop[idx] = self.generate_agent()
        # Damp Dance Radius
        self.dyn_radius = self.dance_reduction * self.dance_radius
