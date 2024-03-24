#!/usr/bin/env python
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevQSA(Optimizer):
    """
    The developed version: Queuing Search Algorithm (QSA)

    Notes:
        + The third loops are removed
        + Global best solution is used in business 3-th instead of random solution

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, QSA
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
    >>> model = QSA.DevQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
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

    def calculate_queue_length__(self, t1, t2, t3):
        """
        Calculate length of each queue based on  t1, t2,t3
            + t1 = t1 * 1.0e+100
            + t2 = t2 * 1.0e+100
            + t3 = t3 * 1.0e+100
        """
        if t1 > 1.0e-6:
            n1 = (1 / t1) / ((1 / t1) + (1 / t2) + (1 / t3))
            n2 = (1 / t2) / ((1 / t1) + (1 / t2) + (1 / t3))
        else:
            n1 = 1.0 / 3
            n2 = 1.0 / 3
        q1 = int(n1 * self.pop_size)
        q2 = int(n2 * self.pop_size)
        q3 = self.pop_size - q1 - q2
        return q1, q2, q3

    def update_business_1__(self, pop=None, current_epoch=None):
        A1, A2, A3 = pop[0].solution, pop[1].solution, pop[2].solution
        t1, t2, t3 = pop[0].target.fitness, pop[1].target.fitness, pop[2].target.fitness
        q1, q2, q3 = self.calculate_queue_length__(t1, t2, t3)
        case = None
        for idx in range(self.pop_size):
            if idx < q1:
                if idx == 0:
                    case = 1
                A = A1.copy()
            elif q1 <= idx < q1 + q2:
                if idx == q1:
                    case = 1
                A = A2.copy()
            else:
                if idx == q1 + q2:
                    case = 1
                A = A3.copy()
            beta = np.power(current_epoch, np.power(current_epoch / self.epoch, 0.5))
            alpha = self.generator.uniform(-1, 1)
            E = self.generator.exponential(0.5, self.problem.n_dims)
            F1 = beta * alpha * (E * np.abs(A - pop[idx].solution)) + self.generator.exponential(0.5) * (A - pop[idx].solution)
            F2 = beta * alpha * (E * np.abs(A - pop[idx].solution))
            if case == 1:
                pos_new = A + F1
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, pop[idx].target, self.problem.minmax):
                    pop[idx] = agent
                else:
                    case = 2
            else:
                pos_new = pop[idx].solution + F2
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, pop[idx].target, self.problem.minmax):
                    pop[idx] = agent
                else:
                    case = 1
        return self.get_sorted_population(pop, self.problem.minmax)

    def update_business_2__(self, pop=None):
        A1, A2, A3 = pop[0].solution, pop[1].solution, pop[2].solution
        t1, t2, t3 = pop[0].target.fitness, pop[1].target.fitness, pop[2].target.fitness
        q1, q2, q3 = self.calculate_queue_length__(t1, t2, t3)
        pr = [idx / self.pop_size for idx in range(1, self.pop_size + 1)]
        if t1 > 1.0e-005:
            cv = t1 / (t2 + t3)
        else:
            cv = 1.0 / 2
        pop_new = []
        for idx in range(self.pop_size):
            if idx < q1:
                A = A1.copy()
            elif q1 <= idx < q1 + q2:
                A = A2.copy()
            else:
                A = A3.copy()
            if self.generator.random() < pr[idx]:
                i1, i2 = self.generator.choice(self.pop_size, 2, replace=False)
                if self.generator.random() < cv:
                    X_new = pop[idx].solution + self.generator.exponential(0.5) * (pop[i1].solution - pop[i2].solution)
                else:
                    X_new = pop[idx].solution + self.generator.exponential(0.5) * (A - pop[i1].solution)
            else:
                X_new = self.problem.generate_solution()
            pos_new = self.correct_solution(X_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_new[-1] = self.get_better_agent(agent, pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            pop_new = self.greedy_selection_population(pop, pop_new, self.problem.minmax)
        return self.get_sorted_and_trimmed_population(pop_new, self.pop_size, self.problem.minmax)

    def update_business_3__(self, pop, g_best):
        pr = np.array([idx / self.pop_size for idx in range(1, self.pop_size + 1)])
        pop_new = []
        for idx in range(self.pop_size):
            X_new = pop[idx].solution.copy()
            id1 = self.generator.choice(self.pop_size)
            temp = g_best.solution + self.generator.exponential(0.5, self.problem.n_dims) * (pop[id1].solution - pop[idx].solution)
            X_new = np.where(self.generator.random(self.problem.n_dims) > pr[idx], temp, X_new)
            pos_new = self.correct_solution(X_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_new[-1] = self.get_better_agent(agent, pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            pop_new = self.greedy_selection_population(pop, pop_new, self.problem.minmax)
        return pop_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.pop, epoch)
        pop = self.update_business_2__(pop)
        self.pop = self.update_business_3__(pop, self.g_best)


class OppoQSA(DevQSA):
    """
    The opposition-based learning version: Queuing Search Algorithm (OQSA)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, QSA
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
    >>> model = QSA.OppoQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = True

    def opposition_based__(self, pop = None, g_best = None):
        pop = self.get_sorted_population(pop, self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.generate_opposition_solution(pop[idx], g_best)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_new[-1] = self.get_better_agent(agent, pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            pop_new = self.greedy_selection_population(pop, pop_new, self.problem.minmax)
        return pop_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.pop, epoch)
        pop = self.update_business_2__(pop)
        pop = self.update_business_3__(pop, self.g_best)
        self.pop = self.opposition_based__(pop, self.g_best)


class LevyQSA(DevQSA):
    """
    The Levy-flight version: Queuing Search Algorithm (LQSA)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, QSA
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
    >>> model = QSA.LevyQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = True

    def update_business_2__(self, pop=None, current_epoch=None):
        A1, A2, A3 = pop[0].solution, pop[1].solution, pop[2].solution
        t1, t2, t3 = pop[0].target.fitness, pop[1].target.fitness, pop[2].target.fitness
        q1, q2, q3 = self.calculate_queue_length__(t1, t2, t3)
        pr = [idx / self.pop_size for idx in range(1, self.pop_size + 1)]
        if t1 > 1.0e-6:
            cv = t1 / (t2 + t3)
        else:
            cv = 1 / 2
        pop_new = []
        for idx in range(self.pop_size):
            if idx < q1:
                A = A1.copy()
            elif q1 <= idx < q1 + q2:
                A = A2.copy()
            else:
                A = A3.copy()
            if self.generator.random() < pr[idx]:
                id1 = self.generator.choice(self.pop_size)
                if self.generator.random() < cv:
                    levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.001, case=-1)
                    X_new = pop[idx].solution + self.generator.normal(0, 1, self.problem.n_dims) * levy_step
                else:
                    X_new = pop[idx].solution + self.generator.exponential(0.5) * (A - pop[id1].solution)
                pos_new = self.correct_solution(X_new)
            else:
                pos_new = self.problem.generate_solution()
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_new[-1] = self.get_better_agent(agent, pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            pop_new = self.greedy_selection_population(pop, pop_new, self.problem.minmax)
        return self.get_sorted_and_trimmed_population(pop_new, self.pop_size, self.problem.minmax)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.pop, epoch)
        pop = self.update_business_2__(pop, epoch)
        self.pop = self.update_business_3__(pop, self.g_best)


class ImprovedQSA(OppoQSA, LevyQSA):
    """
    The original version of: Improved Queuing Search Algorithm (QSA)

    Links:
       1. https://doi.org/10.1007/s12652-020-02849-4

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, QSA
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
    >>> model = QSA.ImprovedQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, B.M., Hoang, B., Nguyen, T. and Nguyen, G., 2021. nQSV-Net: a novel queuing search variant for
    global space search and workload modeling. Journal of Ambient Intelligence and Humanized Computing, 12(1), pp.27-46.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.pop, epoch)
        pop = self.update_business_2__(pop, epoch)
        pop = self.update_business_3__(pop, self.g_best)
        self.pop = self.opposition_based__(pop, self.g_best)


class OriginalQSA(DevQSA):
    """
    The original version of: Queuing Search Algorithm (QSA)

    Links:
       1. https://www.sciencedirect.com/science/article/abs/pii/S0307904X18302890

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, QSA
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
    >>> model = QSA.OriginalQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhang, J., Xiao, M., Gao, L. and Pan, Q., 2018. Queuing search algorithm: A novel metaheuristic algorithm
    for solving engineering optimization problems. Applied Mathematical Modelling, 63, pp.464-490.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = True

    def update_business_3__(self, pop, g_best):
        pr = [idx / self.pop_size for idx in range(1, self.pop_size + 1)]
        pop_new = []
        for idx in range(self.pop_size):
            pos_new = pop[idx].solution.copy()
            for jdx in range(self.problem.n_dims):
                if self.generator.random() > pr[idx]:
                    i1, i2 = self.generator.choice(self.pop_size, 2, replace=False)
                    e = self.generator.exponential(0.5)
                    X1 = pop[i1].solution
                    X2 = pop[i2].solution
                    pos_new[jdx] = X1[jdx] + e * (X2[jdx] - pop[idx].solution[jdx])
            pos_new= self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_new[-1] = self.get_better_agent(agent, pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            pop_new = self.greedy_selection_population(pop, pop_new, self.problem.minmax)
        return pop_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.pop, epoch)
        pop = self.update_business_2__(pop)
        self.pop = self.update_business_3__(pop, self.g_best)
