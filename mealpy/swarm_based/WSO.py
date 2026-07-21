#!/usr/bin/env python
# Created by "Thieu" at 22:35, 16/07/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalWSO(Optimizer):
    """
    The original version: White Shark Optimizer (WSO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Population size, in range [5, 100000]. Default is 100.
    tau : float
        Acceleration coefficient used to derive the constriction factor `mu`, in range [0.0, 100.0]. Default is 4.125.
    p_min : float
        Initial velocities to control the effect of global and local best positions, in range [0.0, 10.0]. Default is 0.5.
    p_max : float
        Subordinate velocities to control the effect of global and local best positions, in range [0.0, 100.0]. Default is 1.5.
    f_min : float
        Minimum frequencies of the undulating motion, in range (0.0, 10.0). Default is 0.07.
    f_max : float
        Maximum frequencies of the undulating motion, in range (0.0, 10.0). Default is 0.75.
    a0 : float
        Constant managing exploration vs. exploitation via the movement force parameter `mv` (hearing/smell strength), in range (0.0, 1000.0). Default is 6.25.
    a1 : float
        Constant managing exploration vs. exploitation via the movement force parameter `mv` (hearing/smell strength), in range (0.0, 1000.0). Default is 100.0.
    a2 : float
        Constant controlling the sight/smell strength when following the best shark in the school (`s_s`), in range (0.0, 1000.0). Default is 0.0005.

    Warnings
    --------
    1. Discrepancies have been spotted between the MATLAB code and the pseudocode presented in
       the algorithm's paper. Users should exercise caution when using this algorithm.
    2. This version accurately implements the equations from the paper, allowing users to
       validate both the algorithm's performance and the published results.
    3. A drawback of this algorithm is the introduction of too many meaningless parameters. Replacing them
       with simpler operators could potentially improve performance while eliminating the need for parameter tuning
    4. Many parameters are fixed in the paper, but this heavily depends on your specific problem. Therefore,
       users are advised to read the paper carefully to understand the functional meaning of these hyperparameters.

    Links
    -----
    1. https://doi.org/10.1016/j.knosys.2022.108457
    2. https://www.mathworks.com/matlabcentral/fileexchange/107365-white-shark-optimizer-wso

    References
    ----------
    1. Braik, M., Hammouri, A., Atwan, J., Al-Betar, M. A., & Awadallah, M. A. (2022).
       White Shark Optimizer: A novel bio-inspired meta-heuristic algorithm for global optimization
       problems. Knowledge-Based Systems, 243, 108457.

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, WSO
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
    >>> model = WSO.OriginalWSO(epoch=1000, pop_size=50, tau=4.2, p_min=0.5, p_max=2.0, f_min=0.1, f_max=0.8, a0=6, a1=100, a2=0.001)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, tau: float=4.125, p_min: float=0.5, p_max: float=1.5,
                 f_min: float=0.07, f_max: float=0.75, a0: float=6.25, a1: float=100.0, a2: float=0.0005, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 100000])
        self.tau = self.validator.check_float("tau", tau, [0, 100.])
        self.p_min = self.validator.check_float("p_min", p_min, [0, 10.])
        self.p_max = self.validator.check_float("p_max", p_max, [0, 100.])
        self.f_min = self.validator.check_float("f_min", f_min, (0, 10.))
        self.f_max = self.validator.check_float("f_max", f_max, (0, 10.))
        self.a0 = self.validator.check_float("a0", a0, (0, 1000.))
        self.a1 = self.validator.check_float("a1", a1, (0, 1000.))
        self.a2 = self.validator.check_float("a2", a2, (0, 1000.))
        self.set_parameters(["epoch", "pop_size", "tau", "p_min", "p_max", "f_min", "f_max", "a0", "a1", "a2"])
        self.sort_flag = False
        self.v = None

    def before_main_loop(self):
        self.v = np.zeros((self.pop_size, self.problem.n_dims))
        # Pre-calculate the constriction factor mu (Eq. 9)
        self.mu = 2.0 / abs(2.0 - self.tau - np.sqrt(self.tau ** 2 - 4.0 * self.tau))

    def evolve(self, epoch):
        """
        The main evolution step.
        """
        # Dynamically update optimization control parameters as functions of current iteration
        p1 = self.p_max + (self.p_max - self.p_min) * np.exp(-((4.0 * epoch / self.epoch) ** 2))
        p2 = self.p_min + (self.p_max - self.p_min) * np.exp(-((4.0 * epoch / self.epoch) ** 2))

        # mv: Movement force representing hearing and smell strength (Eq. 15)
        mv = 1.0 / (self.a0 + np.exp((self.epoch / 2.0 - epoch) / self.a1))

        # s_s: Senses of smell and sight for fish school tracking behavior (Eq. 18)
        s_s = abs(1.0 - np.exp(-self.a2 * epoch / self.epoch))

        # Identify the best known position vector known to the swarm (v_index) (Eq. 6)
        pos_list = np.array([agent.solution for agent in self.pop])
        v_idx = np.floor(self.pop_size * self.generator.uniform(0, 1, self.pop_size)).astype(int)
        w_best_v = pos_list[v_idx]

        c1 = self.generator.uniform(0, 1, (self.pop_size, self.problem.n_dims))
        c2 = self.generator.uniform(0, 1, (self.pop_size, self.problem.n_dims))

        # Velocity update formulation (Eq. 5)
        v = self.mu * (self.v + p1 * (self.g_best.solution - pos_list) * c1 + p2 * (w_best_v - pos_list) * c2)

        # Generate wave frequencies for undulating motion (Eq. 14)
        f = self.f_min + (self.f_max - self.f_min) * self.generator.uniform(0, 1, self.pop_size)

        w_new = np.zeros_like(pos_list)
        rand_vals = self.generator.uniform(0, 1, self.pop_size)
        for idx in range(self.pop_size):
            # Position update Step 1: Movement towards prey (Eq. 10)
            if rand_vals[idx] < mv:
                # Random exploration bounding vectors
                a = (pos_list[idx] - self.problem.ub > 0).astype(int)
                b = (pos_list[idx] - self.problem.lb < 0).astype(int)
                w_o = np.bitwise_xor(a, b)
                # Apply logical mapping for random target tracking around the optimal prey
                w_new[idx] = pos_list[idx] * np.logical_not(w_o) + self.problem.ub * a + self.problem.lb * b
            else:
                # Move towards prey using undulating wavy motion
                w_new[idx] = pos_list[idx] + v[idx] / f[idx]

        # Position update Step 2: Fish school behavior and movement towards the best shark
        for idx in range(self.pop_size):
            # Ensure intermediate position stays within boundaries before applying collective behavior
            w_new[idx] = np.clip(w_new[idx], self.problem.lb, self.problem.ub)

            # Calculate distance between prey and white shark (Eq. 17)
            d_w = abs(self.generator.uniform(0, 1, self.problem.n_dims) * (self.g_best.solution - w_new[idx]))
            r1 = self.generator.uniform(0, 1, self.problem.n_dims)
            r2 = self.generator.uniform(0, 1, self.problem.n_dims)

            # Emulate collective behavior near optimal target (Eq. 16, 19)
            if idx < s_s * self.pop_size:
                sgn = np.where(r2 - 0.5 > 0, 1, -1)
                w_hat = self.g_best.solution + r1 * d_w * sgn
                # Position update respecting the fish school consensus (Eq. 19)
                w_new[idx] = (w_new[idx] + w_hat) / (2.0 * self.generator.uniform(0, 1, self.problem.n_dims))

        # Boundary and update agent
        pop_new = []
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(w_new[idx])
            # agent = self.generate_empty_agent(pos_new)
            pop_new.append(self.generate_empty_agent(pos_new))
            # Update fitness in single mode
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(pop_new[-1], self.pop[idx], self.problem.minmax)
        # Update fitness in parallel modes
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
