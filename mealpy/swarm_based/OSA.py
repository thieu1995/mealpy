#!/usr/bin/env python
# Created by "Furkan Buyukyozgat" at 15:25, 05/01/2026-------%
#       Email: furkanbuyuky@gmail.com                        %
#       Github: https://github.com/furkanbuyuky              %
# -----------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalOSA(Optimizer):
    """
    The original version: Owl Search Algorithm (OSA)

    Warnings
    --------
    There are two MATLAB versions of this algorithm available; however, neither is from the original authors,
    and their implementations do not accurately reflect the original paper. This algorithm was published
    in a low-tier journal, lacks any unique update operators, and does not provide pseudocode,
    which explains why it hasn't gained traction since its publication in 2018.

    Links
    -----
    1. https://www.mathworks.com/matlabcentral/fileexchange/181126-owl-search-algorithm-osa
    2. https://www.mathworks.com/matlabcentral/fileexchange/162356-owl-search-algorithm-osa

    Hyperparameters
    ----------------
    + alpha_max (float): (0, 1.0), alpha max step size, default=0.5
    + beta_max (float): (0, 2.0), beta max step size, default=1.9

    References
    ----------
    .. [1] Jain, M., Maurya, S., Rani, A., & Singh, V. (2018). Owl search algorithm: a novel nature-inspired
    heuristic paradigm for global optimization. Journal of Intelligent & Fuzzy Systems, 34(3), 1573-1582.
    https://doi.org/10.3233/JIFS-169452

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, OSA
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
    >>> model = OSA.OriginalOSA(epoch=1000, pop_size=50, alpha_max = 0.5, beta_max = 1.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, alpha_max: float = 0.5, beta_max: float = 1.9, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 100000])
        self.alpha_max = self.validator.check_float("alpha_max", alpha_max, (0.0, 10.0))
        self.beta_max = self.validator.check_float("beta_max", beta_max, (0.0, 10.0))
        self.set_parameters(["epoch", "pop_size", "beta_max", "alpha_max"])
        self.sort_flag = False

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        alpha = self.generator.random() * self.alpha_max
        beta = self.beta_max * (1.0 - epoch / self.epoch)

        fits = np.array([agent.target.fitness for agent in self.pop])
        fmax = np.max(fits)
        fmin = np.min(fits)
        if self.problem.minmax == "min":
            best_fit, worst_fit = fmin, fmax
        else:
            best_fit, worst_fit = fmax, fmin
        if abs(best_fit - worst_fit) < self.EPSILON:
            I_i = np.ones(self.pop_size)
        else:
            I_i = (fits - worst_fit) / (best_fit - worst_fit)

        # Equation (7): Distance calculation R_i = ||O_i, V||_2
        # Epsilon (1e-9) is added to prevent division by zero for the best owl
        owls = np.array([agent.solution for agent in self.pop])
        R_i = np.linalg.norm(owls - self.g_best.solution, axis=1) + self.EPSILON

        # Equation (8): Change in intensity Ic_i obeying inverse square law
        Ic_i = (I_i / (R_i ** 2)) + self.generator.random(self.pop_size)

        for idx in range(self.pop_size):
            p_vm = self.generator.random()
            # Step size magnitude
            step = beta * Ic_i[idx] * (alpha * self.g_best.solution - owls[idx])

            # Prey movement probability trigger
            if p_vm < 0.5:
                pos_new = owls[idx] + step
            else:
                pos_new = owls[idx] - step

            # Apply boundary constraints
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            self.pop[idx] = agent
            # Update fitness in single mode
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].target = self.get_target(pos_new)
        # Update fitness in parallel modes
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(self.pop)
