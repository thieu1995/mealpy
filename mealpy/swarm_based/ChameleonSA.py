#!/usr/bin/env python
# Created by "Ulaş Görkem Kazan" on 05/01/2026
# Github: https://github.com/gorkemulas2005
# --------------------------------------------------%
# Updated by "Thieu" on 16/07/2026
# Github: https://github.com/thieu1995
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalChameleonSA(Optimizer):
    """
    The original version of: Chameleon Swarm Algorithm (ChameleonSA)

    Hyperparameters
    ---------------
    + epoch (int): Maximum number of iterations, default = 10000
    + pop_size (int): Population size, default = 100
    + pp (float): [0, 1] Probability of the chameleon perceiving prey, default=0.1
    + p1 (float): [0, 5.0] Exploration control parameter 1 (From PSO), default=0.25
    + p2 (float): [0, 5.0] Exploration control parameter 2 (From PSO), (default=1.50)
    + c1 (float): [0, 5.0] Personal best influence (From PSO), default=1.75
    + c2 (float): [0, 5.0] Global best influence (From PSO), default=1.75
    + gama (float): [0, 2] Constant controlling the exploration rate decay over iterations, default=1.0
    + alpha (float): [0, 10] Constant defining the steepness of the exploration decay curve, default=3.5
    + rho (float): [0, 2] Positive number, default=1.0.

    Warnings
    --------
    1. This algorithm essentially relies on the update operators of the PSO algorithm. It has too many
    parameters, and the results are nowhere near as good as those presented in the paper
    2. Please note that the official MATLAB code deviates from the paper, using undocumented
    modifications to artificially boost performance.
    3. This pure implementation is provided specifically so users can independently evaluate
    the algorithm's true performance based solely on the published mathematical model,
    allowing you to verify whether the paper's claims and results are legitimate.

    Links
    -----
    1. https://www.mathworks.com/matlabcentral/fileexchange/98014-chameleon-swarm-algorithm
    2. https://doi.org/10.1016/j.eswa.2021.114685

    References
    ----------
    .. [1] Braik, M. S. (2021). Chameleon Swarm Algorithm: A bio-inspired optimizer for solving
    engineering design problems. Expert Systems with Applications, 174, 114685.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ChameleonSA
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
    >>> model = ChameleonSA.OriginalChameleonSA(epoch=1000, pop_size=50, pp=0.2, p1=0.3, p2=2.0, c1=2.0, c2=2.0, gama=1.0, alpha=5.0, rho=1.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, pp: float=0.1, p1: float=0.25, p2: float=1.50,
                 c1: float=1.75, c2: float=1.75, gama: float=1.0, alpha: float=3.5, rho: float=1.0, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 100000])
        self.pp = self.validator.check_float("pp", pp, [0, 1])
        self.p1 = self.validator.check_float("p1", p1, [0, 10.])
        self.p2 = self.validator.check_float("p2", p2, [0, 10.])
        self.c1 = self.validator.check_float("c1", c1, [0, 10.])
        self.c2 = self.validator.check_float("c2", c2, [0, 10.])
        self.gama = self.validator.check_float("gama", gama, [0, 10.])
        self.alpha = self.validator.check_float("alpha", alpha, [0, 10.])
        self.rho = self.validator.check_float("rho", rho, [0, 10.])

        self.set_parameters(["epoch", "pop_size", "pp", "p1", "p2", "c1", "c2", "gama", "alpha", "rho"])
        self.sort_flag = True
        self.pop_personal = None
        self.velocities = None
        self.prev_velocities = None

    def before_main_loop(self):
        self.pop_personal = self.pop.copy()
        self.velocities = np.zeros((self.pop_size, self.problem.n_dims))
        self.prev_velocities = np.zeros((self.pop_size, self.problem.n_dims))

    def evolve(self, epoch):
        """
        The main evolution step.
        """
        # Dynamic parameters update
        mu = self.gama * np.exp(-self.alpha * epoch / self.epoch)  # Eq. 6
        omega = (1 - epoch / self.epoch) ** (self.rho * np.sqrt(epoch / self.epoch))  # Eq. 19
        a = 2590 * (1 - np.exp(-np.log(epoch + 1)))  # Eq. 21 (acceleration)

        for idx in range(self.pop_size):
            # Phase 1: Search for prey (Eq. 3)
            rr = self.generator.random()
            r1, r2, r3 = self.generator.random(3)
            if rr >= self.pp:
                pos_new = self.pop[idx].solution + self.p1 * (self.pop_personal[idx].solution - self.g_best.solution) * r1 + \
                          self.p2 * (self.g_best.solution - self.pop[idx].solution) * r2
            else:
                direction = np.sign(self.generator.random(self.problem.n_dims) - 0.5)
                pos_new = self.pop[idx].solution + mu * (((self.problem.ub - self.problem.lb) * r3 + self.problem.lb) * direction)
            self.pop[idx].solution = pos_new

        center = np.mean([agent.solution for agent in self.pop], axis=0)
        for idx in range(self.pop_size):
            # Phase 2: Chameleon eyes rotation (Eq. 11 - 15)
            yc = self.pop[idx].solution - center  # Eq. 13
            theta = self.generator.random(self.problem.n_dims) * np.sign(self.generator.random(self.problem.n_dims) - 0.5) * np.pi  # Eq. 15
            # Apply rotation (Eq. 12)
            yr = np.cos(theta) * yc
            self.pop[idx].solution = yr + center  # Eq. 11

        for idx in range(self.pop_size):
            # Phase 3: Hunting prey - Tongue projection (Eq. 18 - 20)
            r1, r2 = self.generator.random(2)
            self.velocities[idx] = omega * self.velocities[idx] + \
                            self.c1 * r1 * (self.g_best.solution - self.pop[idx].solution) + \
                            self.c2 * r2 * (self.pop_personal[idx].solution - self.pop[idx].solution)
            delta_v_squared = (self.velocities[idx] ** 2 - self.prev_velocities[idx] ** 2)
            tongue_step = delta_v_squared / (2 * (a + self.EPSILON))
            self.pop[idx].solution = self.pop[idx].solution + tongue_step
            self.prev_velocities[idx] = self.velocities[idx].copy()

        # Adjust boundaries (Line 32)
        for idx in range(self.pop_size):
            self.pop[idx].solution = self.correct_solution(self.pop[idx].solution)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].target = self.get_target(self.pop[idx].solution)
        # Update fitness in parallel modes
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(self.pop)

        # Update personal best positions
        for idx in range(self.pop_size):
            if self.compare_target(self.pop[idx].target, self.pop_personal[idx].target, self.problem.minmax):
                self.pop_personal[idx] = self.pop[idx].copy()


class IChameleonSA(Optimizer):
    """
    The original version of: Improved Chameleon Swarm Algorithm (ICSA)

    Hyperparameters
    ---------------
    + epoch (int): Maximum number of iterations, default = 10000
    + pop_size (int): Population size, default = 100
    + beta (float): Lévy flight constant (Eq. 13), default = 1.5
    + r_chaos (float): Control parameter for logistic mapping (Eq. 10), default = 0.3
    + k_spiral (int): Variation coefficient for spiral search (Eq. 11), default = 5
    + p1 (float): [0, 10.] Personal best influence (From PSO), default=2.0
    + p2 (float): [0, 10.] Global best influence (From PSO), default=2.0

    Warnings
    --------
    1. Despite being claimed as an improved version, this algorithm still requires too many
    parameters and relies on standard PSO update operators
    2. Additionally, its NFE per iteration is 3x times higher than typical algorithms,
    so users should be mindful of the execution time.

    References
    ----------
    .. [1] Chen, Yaodan, Li Cao, and Yinggao Yue. "Hybrid Multi-Objective Chameleon Optimization Algorithm
    Based on Multi-Strategy Fusion and Its Applications." Biomimetics 9.10 (2024): 583.
    https://doi.org/10.3390/biomimetics9100583

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ChameleonSA
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
    >>> model = ChameleonSA.IChameleonSA(epoch=1000, pop_size=50, r_chaos=0.5, k_spiral=10., p1=5.0, p2=3.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    def __init__(self, epoch=1000, pop_size=100, r_chaos: float=0.3, k_spiral: float=5.0, p1: float=2.0, p2: float=2.0, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.r_chaos = self.validator.check_float("r_chaos", r_chaos, [0.0, 1.0])
        self.k_spiral = self.validator.check_float("k_spiral", k_spiral, [0., 100.0])
        self.p1 = self.validator.check_float("p1", p1, [0., 100.0])
        self.p2 = self.validator.check_float("p2", p2, [0., 100.0])
        self.set_parameters(["epoch", "pop_size", "r_chaos", "k_spiral", "p1", "p2"])
        self.sort_flag = False
        self.V = self.pop_personal = None

    def initialization(self) -> None:
        if self.pop is None:
            # 4.1. Logistic Chaotic Map Initialization (Eq. 9 & 10)
            l_seq = self.generator.random((self.pop_size, self.problem.n_dims))
            for j in range(self.pop_size - 1):
                l_seq[j + 1] = self.r_chaos * l_seq[j] * (1 - l_seq[j])  # Eq. (10)
            pop_pos = self.problem.lb + l_seq * (self.problem.ub - self.problem.lb)  # Eq. (9)
            self.pop = []
            for idx in range(self.pop_size):
                pos_new = self.correct_solution(pop_pos[idx])
                self.pop.append(self.generate_empty_agent(pos_new))
                if self.mode not in self.AVAILABLE_MODES:
                    self.pop[idx].target = self.get_target(pos_new)
            # Update fitness in parallel modes
            if self.mode in self.AVAILABLE_MODES:
                self.pop = self.update_target_for_population(self.pop)
        self.V = np.zeros((self.pop_size, self.problem.n_dims))
        self.pop_personal = self.pop.copy()

    def evolve(self, epoch):
        mu = np.exp(-(3.5 * epoch/ self.epoch) ** 3)  # Eq. (2)

        # Phase 1: Search for prey
        mean_fit = np.mean([agent.target.fitness for agent in self.pop])
        for idx in range(self.pop_size):
            if self.compare_fitness(self.pop[idx].target.fitness, mean_fit, self.problem.minmax):
                r1, r2 = self.generator.random(2)
                pos_new = self.pop[idx].solution + self.p1 * r1 * (self.pop_personal[idx].solution - self.g_best.solution) + \
                          self.p2 * r1 * (self.g_best.solution - self.pop[idx].solution)
            else:
                ll = self.generator.uniform(-1, 1)
                sgn = np.sign(self.generator.random() - 0.5)
                pos_new = self.g_best.solution + np.exp(self.k_spiral * ll) * np.cos(2 * np.pi * ll) * (self.g_best.solution - self.pop[idx].solution) * sgn
            self.pop[idx].solution = pos_new

        # Phase 2: Chameleon eyes' rotation
        c_t = 0.075 * (1 + np.cos((np.pi * epoch) / self.epoch))
        pop_new = []
        for idx in range(self.pop_size):
            # Levy flight step calculation
            levy_step = self.get_levy_flight_step(1.5, multiplier=1, size=self.problem.n_dims, case=-1)
            # Rotation & Levy flight combination (Eq. 15)
            xl = self.pop[idx].solution + c_t * (self.g_best.solution - self.pop[idx].solution) * levy_step
            pos_new = self.correct_solution(xl)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            # Greedy selection (Eq. 16)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        # Parallel mode
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # Phase 3: Hunting prey
        omega_t = (1 - epoch / self.epoch) ** (2 * np.sqrt(epoch / self.epoch))
        lambda_t = (1 - epoch / self.epoch) ** np.sqrt(epoch / self.epoch)
        a = 2590 * (1 - np.exp(-np.log(epoch)))

        for idx in range(self.pop_size):
            # Update velocity (Eq. 18)
            r1, r2 = self.generator.random(2)
            V_new = lambda_t * self.V[idx] + omega_t * r1 * (self.g_best.solution - self.pop[idx].solution) + \
                    omega_t * r2 * (self.pop_personal[idx].solution - self.pop[idx].solution)
            # Update position (Eq. 7)
            if a == 0:
                X_hunt = self.pop[idx].solution
            else:
                X_hunt = self.pop[idx].solution + ((V_new ** 2) - (self.V[idx] ** 2)) / (2 * a)
            self.V[idx] = V_new
            # Refraction reverse learning (Eq. 20)
            X_refract = ((self.problem.lb + self.problem.ub) / 2) + ((self.problem.lb + self.problem.ub) / (2 * epoch)) - (X_hunt / epoch)
            X_hunt = self.correct_solution(X_hunt)
            X_refract = self.correct_solution(X_refract)
            agent_hunt = self.generate_agent(X_hunt)
            agent_refract = self.generate_agent(X_refract)

            # Greedy selection (Eq. 16)
            if self.compare_target(agent_refract.target, agent_hunt.target, self.problem.minmax):
                self.pop[idx] = agent_refract
            else:
                self.pop[idx] = agent_hunt

        # Evaluate fitness and update Personal
        for idx in range(self.pop_size):
            if self.compare_target(self.pop[idx].target, self.pop_personal[idx].target, self.problem.minmax):
                self.pop_personal[idx] = self.pop[idx].copy()
