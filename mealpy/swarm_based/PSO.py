#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalPSO(Optimizer):
    """
    The original version of: Particle Swarm Optimization (PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + w (float): (0., 1.0), Weight min of bird, default = 0.4

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, PSO
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
    >>> model = PSO.OriginalPSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, w=0.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kennedy, J. and Eberhart, R., 1995, November. Particle swarm optimization. In Proceedings of
    ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). IEEE.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, c1: float = 2.05, c2: float = 2.05, w: float = 0.4, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            w_min: Weight min of bird, default = 0.4
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.w = self.validator.check_float("w", w, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "w"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.v_min, self.v_max)
        local_pos = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update weight after each move count  (weight down)
        for idx in range(0, self.pop_size):
            cognitive = self.c1 * self.generator.random(self.problem.n_dims) * (self.pop[idx].local_solution - self.pop[idx].solution)
            social = self.c2 * self.generator.random(self.problem.n_dims) * (self.g_best.solution - self.pop[idx].solution)
            self.pop[idx].velocity = self.w * self.pop[idx].velocity + cognitive + social
            pos_new = self.pop[idx].solution + self.pop[idx].velocity
            pos_new = self.correct_solution(pos_new)
            target = self.get_target(pos_new)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].update(local_solution=pos_new.copy(), local_target=target.copy())


class AIW_PSO(Optimizer):
    """
    The original version of: Adaptive Inertia Weight Particle Swarm Optimization (AIW-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + alpha (float): [0., 1.0], The positive constant, default = 0.4

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, PSO
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
    >>> model = PSO.AIW_PSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, alpha=0.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Qin, Z., Yu, F., Shi, Z., Wang, Y. (2006). Adaptive Inertia Weight Particle Swarm Optimization. In: Rutkowski, L.,
    Tadeusiewicz, R., Zadeh, L.A., Żurada, J.M. (eds) Artificial Intelligence and Soft Computing – ICAISC 2006. ICAISC 2006.
    Lecture Notes in Computer Science(), vol 4029. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11785231_48
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, c1: float = 2.05, c2: float = 2.05, alpha: float = 0.4, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            alpha: The positive constant, default = 0.4
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.alpha = self.validator.check_float("alpha", alpha, [0., 1.0])
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "alpha"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.v_min, self.v_max)
        local_pos = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        for idx in range(0, self.pop_size):
            denom = np.abs(self.pop[idx].local_solution - current_best.solution)
            denom = np.where(denom==0, 1e-6, denom)
            isa = np.abs(self.pop[idx].solution - self.pop[idx].local_solution) / denom     # individual search ability
            w = 1 - self.alpha * (1.0 / (1.0 + np.exp(-isa)))
            cognitive = self.c1 * self.generator.random(self.problem.n_dims) * (self.pop[idx].local_solution - self.pop[idx].solution)
            social = self.c2 * self.generator.random(self.problem.n_dims) * (self.g_best.solution - self.pop[idx].solution)
            velocity = w * self.pop[idx].velocity + cognitive + social
            self.pop[idx].velocity = np.clip(velocity, self.v_min, self.v_max)
            pos_new = self.pop[idx].solution + self.pop[idx].velocity
            pos_new = self.correct_solution(pos_new)
            target = self.get_target(pos_new)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].update(local_solution=pos_new.copy(), local_target=target.copy())


class LDW_PSO(Optimizer):
    """
    The original version of: Linearly Decreasing inertia Weight Particle Swarm Optimization (LDW-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + w_min (float): [0.1, 0.5], Weight min of bird, default = 0.4
        + w_max (float): [0.8, 2.0], Weight max of bird, default = 0.9

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, PSO
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
    >>> model = PSO.LDW_PSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, w_min=0.4, w_max=0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shi, Yuhui, and Russell Eberhart. "A modified particle swarm optimizer." In 1998 IEEE international conference on
    evolutionary computation proceedings. IEEE world congress on computational intelligence (Cat. No. 98TH8360), pp. 69-73. IEEE, 1998.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, c1: float = 2.05, c2: float = 2.05,
                 w_min: float = 0.4, w_max: float = 0.9, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            w_min: Weight min of bird, default = 0.4
            w_max: Weight max of bird, default = 0.9
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.w_min = self.validator.check_float("w_min", w_min, (0, 0.5))
        self.w_max = self.validator.check_float("w_max", w_max, [0.5, 2.0])
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "w_min", "w_max"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.v_min, self.v_max)
        local_pos = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update weight after each move count  (weight down)
        w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
        for idx in range(0, self.pop_size):
            cognitive = self.c1 * self.generator.random(self.problem.n_dims) * (self.pop[idx].local_solution - self.pop[idx].solution)
            social = self.c2 * self.generator.random(self.problem.n_dims) * (self.g_best.solution - self.pop[idx].solution)
            velocity = w * self.pop[idx].velocity + cognitive + social
            self.pop[idx].velocity = np.clip(velocity, self.v_min, self.v_max)
            pos_new = self.pop[idx].solution + self.pop[idx].velocity
            pos_new = self.correct_solution(pos_new)
            target = self.get_target(pos_new)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].update(local_solution=pos_new.copy(), local_target=target.copy())


class P_PSO(Optimizer):
    """
    The original version of: Phasor Particle Swarm Optimization (P-PSO)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, PSO
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
    >>> model = PSO.P_PSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ghasemi, M., Akbari, E., Rahimnejad, A., Razavi, S.E., Ghavidel, S. and Li, L., 2019.
    Phasor particle swarm optimization: a simple and efficient variant of PSO. Soft Computing, 23(19), pp.9701-9718.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.dyn_delta_list = self.generator.uniform(0, 2 * np.pi, self.pop_size)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(-self.v_max, self.v_max)
        local_pos = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            aa = 2 * (np.sin(self.dyn_delta_list[idx]))
            bb = 2 * (np.cos(self.dyn_delta_list[idx]))
            ee = np.abs(np.cos(self.dyn_delta_list[idx])) ** aa
            tt = np.abs(np.sin(self.dyn_delta_list[idx])) ** bb
            v_new = ee * (self.pop[idx].local_solution - self.pop[idx].solution) + tt * (self.g_best.solution - self.pop[idx].solution)
            v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)
            self.pop[idx].velocity = v_new
            pos_new = self.pop[idx].solution + v_new
            pos_new = self.correct_solution(pos_new)
            self.dyn_delta_list[idx] += np.abs(aa + bb) * (2 * np.pi)
            self.v_max = (np.abs(np.cos(self.dyn_delta_list[idx])) ** 2) * (self.problem.ub - self.problem.lb)
            target = self.get_target(pos_new)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].update(local_solution=pos_new.copy(), local_target=target.copy())


class HPSO_TVAC(P_PSO):
    """
    The original version of: Hierarchical PSO Time-Varying Acceleration (HPSO-TVAC)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + ci (float): [0.3, 1.0], c initial, default = 0.5
        + cf (float): [0.0, 0.3], c final, default = 0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, PSO
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
    >>> model = PSO.HPSO_TVAC(epoch=1000, pop_size=50, ci=0.5, cf=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ghasemi, M., Aghaei, J. and Hadipour, M., 2017. New self-organising hierarchical PSO with
    jumping time-varying acceleration coefficients. Electronics Letters, 53(20), pp.1360-1362.
    """

    def __init__(self, epoch=10000, pop_size=100, ci=0.5, cf=0.1, **kwargs):
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            ci: c initial, default = 0.5
            cf: c final, default = 0.0
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.ci = self.validator.check_float("ci", ci, [0.3, 1.0])
        self.cf = self.validator.check_float("cf", cf, [0, 0.3])
        self.set_parameters(["epoch", "pop_size", "ci", "cf"])
        self.sort_flag = False
        self.is_parallelizable = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        c_it = ((self.cf - self.ci) * ((epoch + 1) / self.epoch)) + self.ci
        for idx in range(0, self.pop_size):
            idx_k = self.generator.integers(0, self.pop_size)
            w = self.generator.normal()
            while np.abs(w - 1.0) < 0.01:
                w = self.generator.normal()
            c1_it = np.abs(w) ** (c_it * w)
            c2_it = np.abs(1 - w) ** (c_it / (1 - w))
            #################### HPSO
            v_new = c1_it * self.generator.uniform(0, 1, self.problem.n_dims) * (self.pop[idx].local_solution - self.pop[idx].solution) + \
                    c2_it * self.generator.uniform(0, 1, self.problem.n_dims) * \
                    (self.g_best.solution + self.pop[idx_k].local_solution - 2 * self.pop[idx].solution)
            v_new = np.where(v_new == 0, np.sign(0.5 - self.generator.uniform()) * self.generator.uniform() * self.v_max, v_new)
            v_new = np.sign(v_new) * np.minimum(np.abs(v_new), self.v_max)
            #########################
            v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)
            pos_new = self.pop[idx].solution + v_new
            pos_new = self.correct_solution(pos_new)
            self.pop[idx].velocity = v_new
            target = self.get_target(pos_new)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].update(local_solution=pos_new.copy(), local_target=target.copy())


class C_PSO(P_PSO):
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
    >>> from mealpy import FloatVar, PSO
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
    >>> model = PSO.C_PSO(epoch=1000, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Liu, B., Wang, L., Jin, Y.H., Tang, F. and Huang, D.X., 2005. Improved particle swarm optimization
    combined with chaos. Chaos, Solitons & Fractals, 25(5), pp.1261-1271.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, c1: float = 2.05, c2: float = 2.05,
                 w_min: float = 0.4, w_max: float = 0.9, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient, default = 2.05
            c2: [0-2] global coefficient, default = 2.05
            w_min: Weight min of bird, default = 0.4
            w_max: Weight max of bird, default = 0.9
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.w_min = self.validator.check_float("w_min", w_min, (0, 0.5))
        self.w_max = self.validator.check_float("w_max", w_max, [0.5, 2.0])
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "w_min", "w_max"])
        self.sort_flag = False
        self.is_parallelizable = False
    
    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max
        self.N_CLS = int(self.pop_size / 5)  # Number of chaotic local searches
        self.dyn_lb = self.problem.lb.copy()
        self.dyn_ub = self.problem.ub.copy()

    def get_weights__(self, fit, fit_avg, fit_min):
        temp1 = self.w_min + (self.w_max - self.w_min) * (fit - fit_min) / (fit_avg - fit_min)
        if self.problem.minmax == "min":
            output = temp1 if fit <= fit_avg else self.w_max
        else:
            output = self.w_max if fit <= fit_avg else temp1
        return output

    def bounded_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.clip(solution, self.dyn_lb, self.dyn_ub)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_fits = [agent.target.fitness for agent in self.pop]
        fit_avg = np.mean(list_fits)
        fit_min = np.min(list_fits)
        for idx in range(self.pop_size):
            w = self.get_weights__(self.pop[idx].target.fitness, fit_avg, fit_min)
            v_new = w * self.pop[idx].velocity + self.c1 * self.generator.random() * (self.pop[idx].local_solution - self.pop[idx].solution) + \
                    self.c2 * self.generator.random() * (self.g_best.solution - self.pop[idx].solution)
            v_new = np.clip(v_new, self.v_min, self.v_max)
            x_new = self.pop[idx].solution + v_new
            self.pop[idx].velocity = v_new
            pos_new = self.bounded_solution(x_new)
            pos_new = self.correct_solution(pos_new)
            target = self.get_target(pos_new)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].update(local_solution=pos_new.copy(), local_target=target.copy())

        ## Implement chaostic local search for the best solution
        g_best = self.g_best.copy()
        cx_best_0 = (self.g_best.solution - self.problem.lb) / (self.problem.ub - self.problem.lb)  # Eq. 7
        cx_best_1 = 4 * cx_best_0 * (1 - cx_best_0)  # Eq. 6
        x_best = self.problem.lb + cx_best_1 * (self.problem.ub - self.problem.lb)  # Eq. 8
        x_best = self.correct_solution(x_best)
        target_best = self.get_target(x_best)
        if self.compare_target(target_best, self.g_best.target):
            g_best.update(solution=x_best, target=target_best)

        r = self.generator.random()
        bound_min = np.stack([self.dyn_lb, g_best.solution - r * (self.dyn_ub - self.dyn_lb)])
        self.dyn_lb = np.max(bound_min, axis=0)
        bound_max = np.stack([self.dyn_ub, g_best.solution + r * (self.dyn_ub - self.dyn_lb)])
        self.dyn_ub = np.min(bound_max, axis=0)

        pop_new_child = self.generate_population(self.pop_size - self.N_CLS)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new_child, self.pop_size, self.problem.minmax)


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
    >>> from mealpy import FloatVar, PSO
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
    >>> model = PSO.CL_PSO(epoch=1000, pop_size=50, c_local = 1.2, w_min=0.4, w_max=0.9, max_flag = 7)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006. Comprehensive learning particle swarm optimizer
    for global optimization of multimodal functions. IEEE transactions on evolutionary computation, 10(3), pp.281-295.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, c_local: float = 1.2,
                 w_min: float = 0.4, w_max: float = 0.9, max_flag: int = 7, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c_local: local coefficient, default = 1.2
            w_min: Weight min of bird, default = 0.4
            w_max: Weight max of bird, default = 0.9
            max_flag: Number of times, default = 7
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c_local = self.validator.check_float("c_local", c_local, (0, 5.0))
        self.w_min = self.validator.check_float("w_min", w_min, (0, 0.5))
        self.w_max = self.validator.check_float("w_max", w_max, [0.5, 2.0])
        self.max_flag = self.validator.check_int("max_flag", max_flag, [2, 100])
        self.set_parameters(["epoch", "pop_size", "c_local", "w_min", "w_max", "max_flag"])
        self.sort_flag = False
        self.is_parallelizable = False
    
    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max
        self.flags = np.zeros(self.pop_size)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(-self.v_max, self.v_max)
        local_pos = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        wk = self.w_max * (epoch / self.epoch) * (self.w_max - self.w_min)
        for idx in range(0, self.pop_size):
            pci = 0.05 + 0.45 * (np.exp(10 * (idx + 1) / self.pop_size) - 1) / (np.exp(10) - 1)
            vec_new = self.pop[idx].velocity.copy()
            for jdx in range(0, self.problem.n_dims):
                if self.generator.random() > pci:
                    vj = wk * self.pop[idx].velocity[jdx] + self.c_local * self.generator.random() * \
                         (self.pop[idx].local_solution[jdx] - self.pop[idx].solution[jdx])
                else:
                    id1, id2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    if self.compare_target(self.pop[id1].target, self.pop[id2].target, self.problem.minmax):
                        vj = wk * self.pop[idx].velocity[jdx] + self.c_local * self.generator.random() * \
                             (self.pop[id1].local_solution[jdx] - self.pop[idx].solution[jdx])
                    else:
                        vj = wk * self.pop[idx].velocity[jdx] + self.c_local * self.generator.random() * \
                             (self.pop[id2].local_solution[jdx] - self.pop[idx].solution[jdx])
                vec_new[jdx] = vj
            vec_new = np.clip(vec_new, self.v_min, self.v_max)
            pos_new = self.pop[idx].solution + vec_new
            pos_new = self.correct_solution(pos_new)
            self.pop[idx].velocity = vec_new
            target = self.get_target(pos_new)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].update(local_solution=pos_new.copy(), local_target=target.copy())
                self.flags[idx] = 0
            else:
                self.flags[idx] += 1
                if self.flags[idx] >= self.max_flag:
                    self.flags[idx] = 0
