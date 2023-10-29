#!/usr/bin/env python
# Created by "Thieu" at 21:58, 16/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalGTO(Optimizer):
    """
    The original version of: Giant Trevally Optimizer (GTO)

    Notes:
        1. This version is implemented exactly as described in the paper.
        2. https://www.mathworks.com/matlabcentral/fileexchange/121358-giant-trevally-optimizer-gto
        3. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9955508

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + A (float): a position-change-controlling parameter with a range from 0.3 to 0.4, default=0.4
        + H (float): initial value for specifies the jumping slope function, default=2.0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GTO
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
    >>> model = GTO.OriginalGTO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic
    Algorithm for Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121615-121640.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, A: float = 0.4, H: float = 2.0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            A (float): a position-change-controlling parameter with a range from 0.3 to 0.4, default=0.4
            H (float): initial value for specifies the jumping slope function, default=2.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.A = self.validator.check_float("A", A, [-10., 10.])
        self.H = self.validator.check_float("H", H, [1., 10.])
        self.set_parameters(["epoch", "pop_size", "A", "H"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Step 1: Extensive Search
        pop_new = []
        for idx in range(0, self.pop_size):
            # Eq.(4)
            pos_new = self.g_best.solution * self.generator.random() + ((self.problem.ub - self.problem.lb) * self.generator.random() + self.problem.lb) * \
                      self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        self.pop, self.g_best = self.update_global_best_agent(self.pop, save=False)

        # Step 2: Choosing Area
        pos_list = np.array([agent.solution for agent in self.pop])
        pos_m = np.mean(pos_list, axis=0)
        pop_new = []
        for idx in range(0, self.pop_size):
            r3 = self.generator.random()
            pos_new = self.g_best.solution * self.A * r3 + pos_m - self.pop[idx].solution * r3        # Eq. 7
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        _, self.g_best = self.update_global_best_agent(self.pop, save=False)

        # Step 3: Attacking
        H = self.generator.random() * (self.H - (epoch+1) * self.H / self.epoch)     #  Eq.(15)
        pop_new = []
        for idx in range(0, self.pop_size):
            # the distance between the prey and the attacker, and can be calculated using (12):
            dist = np.sum(np.abs(self.g_best.solution - self.pop[idx].solution))
            theta2 = (360 - 0) * self.generator.random() + 0
            theta1 = (1.33 / 1.00029) * np.sin(np.radians(theta2))        # calculate theta_1 using (10)
            VD = np.sin(np.radians(theta1)) * dist      # Eq. 11
            # Eq. (13)
            pos_new = self.pop[idx].solution * np.sin(np.radians(theta2)) * self.pop[idx].target.fitness + VD + H
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class Matlab102GTO(Optimizer):
    """
    The conversion of Matlab code (version 1.0.2 - 27/04/2023) to Python code of: Giant Trevally Optimizer (GTO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/121358-giant-trevally-optimizer-gto
        2. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9955508

    Notes:
        1. The authors sent me an email asking to update the algorithm. In this version, they removed 2 for loops in the epoch (generations) based on my comments,
        so the computation time will reduce to 3*pop_size from 2*pop_size^2 + pop_size. However, this will also lead to a reduction in performance results.
        My question: Are the results in the paper valid?
        2. I have decided to implement the original version of the algorithm exactly as described in the paper (OriginalGTO).

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GTO
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
    >>> model = GTO.Matlab102GTO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic
    Algorithm for Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121615-121640.
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
        # Step 1: Extensive Search
        pop_new = []
        for idx in range(0, self.pop_size):
            # foraging movement patterns of giant trevallies are simulated using Eq.(4)
            pos_new = self.g_best.solution * self.generator.random() + ((self.problem.ub - self.problem.lb) * self.generator.random() + self.problem.lb) * \
                    self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        self.pop, self.g_best = self.update_global_best_agent(self.pop, save=False)

        # Step 2: Choosing Area
        A = 0.4
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_m = np.mean(np.array([agent.solution for agent in self.pop]), axis=0)
            # In the choosing area step, giant trevallies identify and select the best area in terms of
            # the amount of food (seabirds) within the selected search space where they can hunt for prey.
            r3 = self.generator.random()
            pos_new = self.g_best.solution * A * r3 + pos_m - self.pop[idx].solution * r3        # Eq. 7

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        self.pop, self.g_best = self.update_global_best_agent(self.pop, save=False)

        # Step 3: Attacking
        H = self.generator.random() * (2.0 - (epoch+1) * 2.0 / self.epoch)     #  Eq.(15)
        pop_new = []
        for idx in range(0, self.pop_size):
            # the distance between the prey and the attacker, and can be calculated using (12):
            dist = np.sum(np.abs(self.g_best.solution - self.pop[idx].solution))
            theta2 = (360 - 0) * self.generator.random() + 0
            theta1 = 1.3296 * np.sin(np.radians(theta2))        # calculate theta_1 using (10)
            # visual distortion indicates the apparent height of the bird, which is always seen
            # to be higher than its actual height due to the refraction of the light.
            VD = np.sin(np.radians(theta1)) * dist      # Eq. 11
            # the behavior of giant trevally when chasing and jumping out of the water is mathematically simulated using (13)
            pos_new = self.pop[idx].solution * np.sin(np.radians(theta2)) * self.pop[idx].target.fitness + VD + H
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class Matlab101GTO(Optimizer):
    """
    The conversion of Matlab code (version 1.0.1 - 29/11/2022) to Python code of: Giant Trevally Optimizer (GTO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/121358-giant-trevally-optimizer-gto
        2. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9955508

    Notes:
        1. This algorithm costs a huge amount of computational resources in each epoch.
        Therefore, be careful when using the maximum number of generations as a stopping condition.
        2. Other algorithms update around K*pop_size times in each epoch, this algorithm updates around 2*pop_size^2 + pop_size times
        3. This version is used by the authors to compared with other algorithms in their paper.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GTO
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
    >>> model = GTO.Matlab101GTO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic
    Algorithm for Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121615-121640.
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
        # Step 1: Extensive Search
        for idx in range(0, self.pop_size):
            pop_new = []
            for jdx in range(0, self.pop_size):
                if idx == jdx:
                    continue
                # foraging movement patterns of giant trevallies are simulated using Eq.(4)
                pos_new = self.g_best.solution * self.generator.random() + ((self.problem.ub - self.problem.lb) * self.generator.random() +
                        self.problem.lb) * self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            pop_new = self.update_target_for_population(pop_new)
            self.pop[idx] = self.get_best_agent(pop_new + [self.pop[idx]], self.problem.minmax)
        _, self.g_best = self.update_global_best_agent(self.pop, save=False)

        # Step 2: Choosing Area
        pos_list = np.array([agent.solution for agent in self.pop])
        pos_m = np.mean(pos_list, axis=0)
        A = 0.4
        pop_new = []
        for idx in range(0, self.pop_size):
            # In the choosing area step, giant trevallies identify and select the best area in terms of
            # the amount of food (seabirds) within the selected search space where they can hunt for prey.
            r3 = self.generator.random()
            pos_new = self.g_best.solution * A * r3 + pos_m - self.pop[idx].solution * r3        # Eq. 7
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        _, self.g_best = self.update_global_best_agent(self.pop, save=False)

        # Step 3: Attacking
        H = self.generator.random() * (2.0 - epoch * 2.0 / self.epoch)     #  Eq.(15)
        for idx in range(0, self.pop_size):
            pop_new = []
            for jdx in range(0, self.pop_size):
                if idx == jdx:
                    continue
                # the distance between the prey and the attacker, and can be calculated using (12):
                dist = np.sum(np.abs(self.g_best.solution - self.pop[idx].solution))
                theta2 = (360 - 0) * self.generator.random() + 0
                theta1 = 1.3296 * np.sin(np.radians(theta2))        # calculate theta_1 using (10)
                # visual distortion indicates the apparent height of the bird, which is always seen
                # to be higher than its actual height due to the refraction of the light.
                VD = np.sin(np.radians(theta1)) * dist      # Eq. 11
                # the behavior of giant trevally when chasing and jumping out of the water is mathematically simulated using (13)
                pos_new = self.pop[idx].solution * np.sin(np.radians(theta2)) * self.pop[idx].target.fitness + VD + H
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            pop_new = self.update_target_for_population(pop_new)
            self.pop[idx] = self.get_best_agent(pop_new + [self.pop[idx]], self.problem.minmax)
