# !/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BasePSO(Optimizer):
    """
    The original version of: Particle Swarm Optimization (PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + w_min (float): [0.1, 0.5], Weight min of bird, default = 0.4
        + w_max (float): [0.8, 2.0], Weight max of bird, default = 0.9

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.PSO import BasePSO
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
    >>> c1 = 2.05
    >>> c2 = 2.05
    >>> w_min = 0.4
    >>> w_max = 0.9
    >>> model = BasePSO(problem_dict1, epoch, pop_size, c1, c2, w_min, w_max)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Kennedy, J. and Eberhart, R., 1995, November. Particle swarm optimization. In Proceedings of
    ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). IEEE.
    """
    ID_POS = 0
    ID_TAR = 1
    ID_VEC = 2  # Velocity
    ID_LOP = 3  # Local position
    ID_LOF = 4  # Local fitness

    def __init__(self, problem, epoch=10000, pop_size=100, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): [0-2] local coefficient
            c2 (float): [0-2] global coefficient
            w_min (float): Weight min of bird, default = 0.4
            w_max (float): Weight max of bird, default = 0.9
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.w_min = self.validator.check_float("w_min", w_min, (0, 0.5))
        self.w_max = self.validator.check_float("w_max", w_max, [0.5, 2.0])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, velocity, local_pos, local_fit]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        velocity = np.random.uniform(self.v_min, self.v_max)
        local_pos = deepcopy(position)
        local_fit = deepcopy(target)
        return [position, target, velocity, local_pos, local_fit]

    def amend_position(self, position=None, lb=None, ub=None):
        """
        Args:
            position: vector position (location) of the solution.
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            Amended position (make the position is in bound)
        """
        condition = np.logical_and(lb <= position, position <= ub)
        pos_rand = np.random.uniform(lb, ub)
        return np.where(condition, position, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update weight after each move count  (weight down)
        w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            v_new = w * self.pop[idx][self.ID_VEC] + self.c1 * np.random.rand() * \
                    (self.pop[idx][self.ID_LOP] - self.pop[idx][self.ID_POS]) + \
                    self.c2 * np.random.rand() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            x_new = self.pop[idx][self.ID_POS] + v_new  # Xi(new) = Xi(old) + Vi(new) * deltaT (deltaT = 1)
            pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            agent[self.ID_POS] = pos_new
            agent[self.ID_VEC] = v_new
            agent[self.ID_TAR] = target
            if self.compare_agent([pos_new, target], [None, self.pop[idx][self.ID_TAR]]):
                agent[self.ID_LOP] = pos_new
                agent[self.ID_LOF] = target
            self.pop[idx] = self.get_better_solution(self.pop[idx], agent)


class PPSO(Optimizer):
    """
    The original version of: Phasor Particle Swarm Optimization (P-PSO)

    Notes
    ~~~~~
    + This code is converted from matlab code (sent from author: Ebrahim Akbari)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.PSO import PPSO
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
    >>> model = PPSO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Ghasemi, M., Akbari, E., Rahimnejad, A., Razavi, S.E., Ghavidel, S. and Li, L., 2019.
    Phasor particle swarm optimization: a simple and efficient variant of PSO. Soft Computing, 23(19), pp.9701-9718.
    """

    ID_VEC = 2  # Velocity
    ID_LOP = 3  # Local position
    ID_LOF = 4  # Local fitness

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

        # Dynamic variable
        self.dyn_delta_list = np.random.uniform(0, 2 * np.pi, self.pop_size)

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, velocity, local_pos, local_fit]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        velocity = np.random.uniform(self.v_min, self.v_max)
        local_pos = deepcopy(position)
        local_fit = deepcopy(target)
        return [position, target, velocity, local_pos, local_fit]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for i in range(0, self.pop_size):
            agent = deepcopy(self.pop[i])
            aa = 2 * (np.sin(self.dyn_delta_list[i]))
            bb = 2 * (np.cos(self.dyn_delta_list[i]))
            ee = np.abs(np.cos(self.dyn_delta_list[i])) ** aa
            tt = np.abs(np.sin(self.dyn_delta_list[i])) ** bb

            v_new = ee * (self.pop[i][self.ID_LOP] - self.pop[i][self.ID_POS]) + tt * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
            v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)
            agent[self.ID_VEC] = deepcopy(v_new)

            pos_new = self.pop[i][self.ID_POS] + v_new
            agent[self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

            self.dyn_delta_list[i] += np.abs(aa + bb) * (2 * np.pi)
            self.v_max = (np.abs(np.cos(self.dyn_delta_list[i])) ** 2) * (self.problem.ub - self.problem.lb)
            pop_new.append(agent)
        # Update fitness for all solutions
        pop_new = self.update_target_wrapper_population(pop_new)

        # Update current position, current velocity and compare with past position, past fitness (local best)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx] = deepcopy(pop_new[idx])
                if self.compare_agent(pop_new[idx], [None, self.pop[idx][self.ID_LOF]]):
                    self.pop[idx][self.ID_LOP] = deepcopy(pop_new[idx][self.ID_POS])
                    self.pop[idx][self.ID_LOF] = deepcopy(pop_new[idx][self.ID_TAR])


class HPSO_TVAC(PPSO):
    """
    The original version of: Ant Colony Optimization Continuous (HPSO-TVAC)

    Notes
    ~~~~~
    + This code is converted from matlab code (sent from author: Ebrahim Akbari)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + ci (float): [0.3, 1.0], c initial, default = 0.5
        + cf (float): [0.0, 0.3], c final, default = 0.0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.PSO import HPSO_TVAC
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
    >>> ci = 0.5
    >>> cf = 0.0
    >>> model = HPSO_TVAC(problem_dict1, epoch, pop_size, ci, cf)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Ghasemi, M., Aghaei, J. and Hadipour, M., 2017. New self-organising hierarchical PSO with
    jumping time-varying acceleration coefficients. Electronics Letters, 53(20), pp.1360-1362.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, ci=0.5, cf=0.0, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ci (float): c initial, default = 0.5
            cf (float): c final, default = 0.0
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.ci = self.validator.check_float("ci", ci, [0.3, 1.0])
        self.cf = self.validator.check_float("cf", cf, [0, 0.3])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        c_it = ((self.cf - self.ci) * ((epoch + 1) / self.epoch)) + self.ci
        pop_new = []
        for i in range(0, self.pop_size):
            agent = deepcopy(self.pop[i])
            idx_k = np.random.randint(0, self.pop_size)
            w = np.random.normal()
            while np.abs(w - 1.0) < 0.01:
                w = np.random.normal()
            c1_it = np.abs(w) ** (c_it * w)
            c2_it = np.abs(1 - w) ** (c_it / (1 - w))

            #################### HPSO
            v_new = c1_it * np.random.uniform(0, 1, self.problem.n_dims) * (self.pop[i][self.ID_LOP] - self.pop[i][self.ID_POS]) + \
                    c2_it * np.random.uniform(0, 1, self.problem.n_dims) * \
                    (self.g_best[self.ID_POS] + self.pop[idx_k][self.ID_LOP] - 2 * self.pop[i][self.ID_POS])

            np.where(v_new == 0, np.sign(0.5 - np.random.uniform()) * np.random.uniform() * self.v_max, v_new)
            v_new = np.sign(v_new) * np.minimum(np.abs(v_new), self.v_max)
            #########################

            v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)
            pos_new = self.pop[i][self.ID_POS] + v_new
            agent[self.ID_VEC] = v_new
            agent[self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append(agent)

            # Update fitness for all solutions
        pop_new = self.update_target_wrapper_population(pop_new)

        # Update current position, current velocity and compare with past position, past fitness (local best)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx] = deepcopy(pop_new[idx])
                if self.compare_agent(pop_new[idx], [None, self.pop[idx][self.ID_LOF]]):
                    self.pop[idx][self.ID_LOP] = deepcopy(pop_new[idx][self.ID_POS])
                    self.pop[idx][self.ID_LOF] = deepcopy(pop_new[idx][self.ID_TAR])


class C_PSO(BasePSO):
    """
    The original version of: Chaos Particle Swarm Optimization (C-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1.0, 3.0] local coefficient, default = 2.05
        + c2 (float): [1.0, 3.0] global coefficient, default = 2.05
        + w_min (float): [0.1, 0.4], Weight min of bird, default = 0.4
        + w_max (float): [0.4, 2.0], Weight max of bird, default = 0.9

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.PSO import C_PSO
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
    >>> c1 = 2.05
    >>> c2 = 2.05
    >>> w_min = 0.4
    >>> w_max = 0.9
    >>> model = C_PSO(problem_dict1, epoch, pop_size, c1, c2, w_min, w_max)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Liu, B., Wang, L., Jin, Y.H., Tang, F. and Huang, D.X., 2005. Improved particle swarm optimization
    combined with chaos. Chaos, Solitons & Fractals, 25(5), pp.1261-1271.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): [0-2] local coefficient, default = 2.05
            c2 (float): [0-2] global coefficient, default = 2.05
            w_min (float): Weight min of bird, default = 0.4
            w_max (float): Weight max of bird, default = 0.9
        """
        super().__init__(problem, epoch, pop_size, c1, c2, w_min, w_max, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.w_min = self.validator.check_float("w_min", w_min, (0, 0.5))
        self.w_max = self.validator.check_float("w_max", w_max, [0.5, 2.0])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max
        self.N_CLS = int(self.pop_size / 5)  # Number of chaotic local searches

        # Dynamic variable
        self.dyn_lb = deepcopy(self.problem.lb)
        self.dyn_ub = deepcopy(self.problem.ub)

    def get_weights__(self, fit, fit_avg, fit_min):
        temp1 = self.w_min + (self.w_max - self.w_min) * (fit - fit_min) / (fit_avg - fit_min)
        if self.problem.minmax == "min":
            output = temp1 if fit <= fit_avg else self.w_max
        else:
            output = self.w_max if fit <= fit_avg else temp1
        return output

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        list_fits = [item[self.ID_TAR][self.ID_FIT] for item in self.pop]
        fit_avg = np.mean(list_fits)
        fit_min = np.min(list_fits)
        pop_new = []
        for i in range(self.pop_size):
            agent = deepcopy(self.pop[i])
            w = self.get_weights__(self.pop[i][self.ID_TAR][self.ID_FIT], fit_avg, fit_min)
            v_new = w * self.pop[i][self.ID_VEC] + self.c1 * np.random.rand() * (self.pop[i][self.ID_LOP] - self.pop[i][self.ID_POS]) + \
                    self.c2 * np.random.rand() * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
            v_new = np.clip(v_new, self.v_min, self.v_max)
            x_new = self.pop[i][self.ID_POS].astype(float) + v_new
            agent[self.ID_VEC] = v_new
            agent[self.ID_POS] = self.amend_position(x_new, self.dyn_lb, self.dyn_ub)
            pop_new.append(agent)

        # Update fitness for all solutions
        pop_new = self.update_target_wrapper_population(pop_new)
        nfe_epoch += self.pop_size

        # Update current position, current velocity and compare with past position, past fitness (local best)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx] = deepcopy(pop_new[idx])
                if self.compare_agent(pop_new[idx], [None, self.pop[idx][self.ID_LOF]]):
                    self.pop[idx][self.ID_LOP] = deepcopy(pop_new[idx][self.ID_POS])
                    self.pop[idx][self.ID_LOF] = deepcopy(pop_new[idx][self.ID_TAR])

        ## Implement chaostic local search for the best solution
        g_best = self.g_best
        cx_best_0 = (self.g_best[self.ID_POS] - self.problem.lb) / (self.problem.ub - self.problem.lb)  # Eq. 7
        cx_best_1 = 4 * cx_best_0 * (1 - cx_best_0)  # Eq. 6
        x_best = self.problem.lb + cx_best_1 * (self.problem.ub - self.problem.lb)  # Eq. 8
        x_best = self.amend_position(x_best, self.problem.lb, self.problem.ub)
        target_best = self.get_target_wrapper(x_best)
        if self.compare_agent([x_best, target_best], self.g_best):
            g_best = [x_best, target_best]

        r = np.random.rand()
        bound_min = np.stack([self.dyn_lb, g_best[self.ID_POS] - r * (self.dyn_ub - self.dyn_lb)])
        self.dyn_lb = np.max(bound_min, axis=0)
        bound_max = np.stack([self.dyn_ub, g_best[self.ID_POS] + r * (self.dyn_ub - self.dyn_lb)])
        self.dyn_ub = np.min(bound_max, axis=0)

        pop_new_child = self.create_population(self.pop_size - self.N_CLS)
        self.pop = self.get_sorted_strim_population(self.pop + pop_new_child, self.pop_size)
        nfe_epoch += 1 + (self.pop_size - self.N_CLS)
        self.nfe_per_epoch = nfe_epoch


class CL_PSO(Optimizer):
    """
    The original version of: Comprehensive Learning Particle Swarm Optimization (CL-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_local (float): [1.0, 3.0], local coefficient, default = 1.2
        + w_min (float): [0.1, 0.5], Weight min of bird, default = 0.4
        + w_max (float): [0.7, 2.0], Weight max of bird, default = 0.9
        + max_flag (int): [5, 20], Number of times, default = 7

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.PSO import CL_PSO
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
    >>> c_local = 1.2
    >>> w_min = 0.4
    >>> w_max = 0.9
    >>> max_flag = 7
    >>> model = CL_PSO(problem_dict1, epoch, pop_size, c_local, w_min, w_max, max_flag)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006. Comprehensive learning particle swarm optimizer
    for global optimization of multimodal functions. IEEE transactions on evolutionary computation, 10(3), pp.281-295.
    """

    ID_VEC = 2
    ID_LOP = 3
    ID_LOF = 4

    def __init__(self, problem, epoch=10000, pop_size=100, c_local=1.2, w_min=0.4, w_max=0.9, max_flag=7, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c_local (float): local coefficient, default = 1.2
            w_min (float): Weight min of bird, default = 0.4
            w_max (float): Weight max of bird, default = 0.9
            max_flag (int): Number of times, default = 7
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c_local = self.validator.check_float("c_local", c_local, (0, 5.0))
        self.w_min = self.validator.check_float("w_min", w_min, (0, 0.5))
        self.w_max = self.validator.check_float("w_max", w_max, [0.5, 2.0])
        self.max_flag = self.validator.check_int("max_flag", max_flag, [2, 100])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

        # Dynamic variable
        self.flags = np.zeros(self.pop_size)

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, velocity, local_pos, local_fit]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        velocity = np.random.uniform(self.v_min, self.v_max)
        local_pos = deepcopy(position)
        local_fit = deepcopy(target)
        return [position, target, velocity, local_pos, local_fit]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        wk = self.w_max * (epoch / self.epoch) * (self.w_max - self.w_min)
        pop_new = []
        for i in range(0, self.pop_size):
            agent = deepcopy(self.pop[i])
            if self.flags[i] >= self.max_flag:
                self.flags[i] = 0
                agent = self.create_solution(self.problem.lb, self.problem.ub)

            pci = 0.05 + 0.45 * (np.exp(10 * (i + 1) / self.pop_size) - 1) / (np.exp(10) - 1)

            vec_new = deepcopy(self.pop[i][self.ID_VEC])
            for j in range(0, self.problem.n_dims):
                if np.random.rand() > pci:
                    vj = wk * self.pop[i][self.ID_VEC][j] + self.c_local * np.random.rand() * \
                         (self.pop[i][self.ID_LOP][j] - self.pop[i][self.ID_POS][j])
                else:
                    id1, id2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                    if self.compare_agent(self.pop[id1], self.pop[id2]):
                        vj = wk * self.pop[i][self.ID_VEC][j] + self.c_local * np.random.rand() * \
                             (self.pop[id1][self.ID_LOP][j] - self.pop[i][self.ID_POS][j])
                    else:
                        vj = wk * self.pop[i][self.ID_VEC][j] + self.c_local * np.random.rand() * \
                             (self.pop[id2][self.ID_LOP][j] - self.pop[i][self.ID_POS][j])
                vec_new[j] = vj
            vec_new = np.clip(vec_new, self.v_min, self.v_max)
            pos_new = self.pop[i][self.ID_POS] + vec_new
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            agent[self.ID_VEC] = vec_new
            agent[self.ID_POS] = pos_new
            pop_new.append(agent)
        pop_new = self.update_target_wrapper_population(pop_new)

        # Update current position, current velocity and compare with past position, past fitness (local best)
        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx] = deepcopy(pop_new[idx])
                if self.compare_agent(pop_new[idx], [None, self.pop[idx][self.ID_LOF]]):
                    self.pop[idx][self.ID_LOP] = deepcopy(pop_new[idx][self.ID_POS])
                    self.pop[idx][self.ID_LOF] = deepcopy(pop_new[idx][self.ID_TAR])
                    self.flags[idx] = 0
                else:
                    self.flags[idx] += 1
