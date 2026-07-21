#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
 
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalWOA(Optimizer):
    """
    The original version of: Whale Optimization Algorithm (WOA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.

    Links
    -----
    1. https://doi.org/10.1016/j.advengsoft.2016.01.008
    2. https://mathworks.com/matlabcentral/fileexchange/55667-the-whale-optimization-algorithm

    References
    ~~~~~~~~~~
    1. Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, WOA
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
    >>> model = WOA.OriginalWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = 2 - 2 * epoch / self.epoch      # linearly decreased from 2 to 0
        a2 = -1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            A = 2 * a * r1 - a
            C = 2 * r2
            bb = 1
            ll = (a2 - 1) * self.generator.random() + 1
            pp = self.generator.random(self.problem.n_dims)

            pos_new = self.pop[idx].solution.copy()
            for jdx in range(0, self.problem.n_dims):
                if pp[jdx] < 0.5:
                    if np.abs(A[jdx]) >= 1:
                        id_r = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                        D_X_rand = abs(C[jdx] * self.pop[id_r].solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.pop[id_r].solution[jdx] - A[jdx] * D_X_rand
                    else:
                        D_Leader = abs(C[jdx] * self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.g_best.solution[jdx] - A[jdx] * D_Leader
                else:
                    D1 = abs(self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                    pos_new[jdx] = D1 * np.exp(bb * ll) * np.cos(ll * 2 * np.pi) + self.g_best.solution[jdx]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].update(solution=pos_new, target=self.get_target(pos_new))
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)


class DevWOA(Optimizer):
    """
    Our developed version of: Whale Optimization Algorithm (WOA)

    Note
    ----
    + Hanlding simple vector instead of loop through whole dimensions
    + Using greedy to update position

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, WOA
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
    >>> model = WOA.DevWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = 2 - 2 * epoch / self.epoch  # linearly decreased from 2 to 0
        pop_new = []
        for idx in range(0, self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            A = 2 * a * r1 - a
            C = 2 * r2
            ll = self.generator.uniform(-1, 1)
            bb = 1

            # Get pos1
            pos1 = self.g_best.solution - A * np.abs(C * self.g_best.solution - self.pop[idx].solution)
            # Get pos2
            id_r2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            pos2 = self.pop[id_r2].solution - A * np.abs(C * self.pop[id_r2].solution - self.pop[idx].solution)
            pos_new = np.where(np.abs(A) < 1, pos1, pos2)

            # Get pos3
            D1 = np.abs(self.g_best.solution - self.pop[idx].solution)
            pos3 = self.g_best.solution + np.exp(bb * ll) * np.cos(2 * np.pi * ll) * D1
            # Get final pos_new
            pos_new = np.where(self.generator.random(size=self.problem.n_dims) < 0.5, pos_new, pos3)

            # Correct solution
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class HI_WOA(Optimizer):
    """
    The original version of: Hybrid Improved Whale Optimization Algorithm (HI-WOA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    feedback_max : int
        Maximum iterations of each feedback, in range [2, 2 + int(epoch/2)]. Default is 10.

    References
    ~~~~~~~~~~
    1. Tang, C., Sun, W., Wu, W. and Xue, M., 2019, July. A hybrid improved whale optimization algorithm.
       In 2019 IEEE 15th International Conference on Control and Automation (ICCA) (pp. 362-367). IEEE.
       https://doi.org/10.1109/ICCA.2019.8900003

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, WOA
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
    >>> model = WOA.HI_WOA(epoch=1000, pop_size=50, feedback_max = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, feedback_max: int = 10, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            feedback_max (int): maximum iterations of each feedback, default = 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.feedback_max = self.validator.check_int("feedback_max", feedback_max, [2, 2+int(self.epoch/2)])
        # The maximum of times g_best doesn't change -> need to change half of population
        self.set_parameters(["epoch", "pop_size", "feedback_max"])
        self.sort_flag = True

    def initialize_variables(self):
        self.n_changes = int(self.pop_size / 2)
        self.dyn_feedback_count = 0

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = 2 + 2 * np.cos(np.pi / 2 * (1 + epoch / self.epoch))  # Eq. 8
        pop_new = []
        for idx in range(0, self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            A = 2 * a * r1 - a
            C = 2 * r2
            ll = self.generator.uniform(-1, 1)
            bb = 1
            if self.generator.uniform() < 0.5:
                # Get pos1
                pos1 = self.g_best.solution - A * np.abs(C * self.g_best.solution - self.pop[idx].solution)
                # Get pos2
                id_r2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                pos2 = self.pop[id_r2].solution - A * np.abs(C * self.pop[id_r2].solution - self.pop[idx].solution)
                pos_new = np.where(np.abs(A) < 1, pos1, pos2)
            else:
                D1 = np.abs(self.g_best.solution - self.pop[idx].solution)
                pos_new = self.g_best.solution + np.exp(bb * ll) * np.cos(2 * np.pi * ll) * D1
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        ## Feedback Mechanism
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        if current_best.target.fitness == self.g_best.target.fitness:
            self.dyn_feedback_count += 1
        else:
            self.dyn_feedback_count = 0

        if self.dyn_feedback_count >= self.feedback_max:
            idx_list = self.generator.choice(range(0, self.pop_size), self.n_changes, replace=False)
            pop_child = self.generate_population(self.n_changes)
            for idx_counter, idx in enumerate(idx_list):
                self.pop[idx] = pop_child[idx_counter]


class OriginalWOAmM(Optimizer):
    """
    The original version of: Whale Optimization Algorithm with Modified Mutualism (WOAmM)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Population size, in range [5, 10000]. Default is 100.
    mut_rand : bool
        Whether mutualism random coefficients are generated per dimension. Default is False.
    patience : int
        Number of stagnant epochs before restarting worst agents. Set 0 to disable, in range [0, 100000]. Default is 0.
    restart_rate : float
        Ratio of worst agents to restart when stagnation occurs, in range [0.0, 1.0]. Default is 0.2.
    bound : str
        Boundary handling method. Supported: "clip", "reflect", "random". Default is "clip".

    References
    ~~~~~~~~~~
    1. Chakraborty, S., Saha, A. K., Sharma, S., Mirjalili, S., & Chakraborty, R. (2021).
       A novel enhanced whale optimization algorithm for global optimization. Computers & Industrial Engineering, 153, 107086.
       https://doi.org/10.1016/j.cie.2020.107086

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, WOA
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
    >>> model = WOA.OriginalWOAmM(epoch=1000, pop_size=50, mut_rand=True, patience=2, restart_rate=0.3, bound="clip")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, mut_rand: bool = False, patience: int = 0,
                 restart_rate: float = 0.2, bound: str = "clip", **kwargs) -> None:
        """
        Args:
            epoch: Maximum number of iterations.
            pop_size: Population size.
            mut_rand: Whether mutualism random coefficients are generated per dimension.
            patience: Number of stagnant epochs before restarting worst agents. Set 0 to disable.
            restart_rate: Ratio of worst agents to restart when stagnation occurs.
            bound: Boundary handling method. Supported: "clip", "reflect", "random".
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mut_rand = self.validator.check_bool("mut_rand", mut_rand)
        self.patience = self.validator.check_int("patience", patience, [0, 100000])
        self.restart_rate = self.validator.check_float("restart_rate", restart_rate, [0.0, 1.0])
        self.bound = self.validator.check_str("bound", bound, ["clip", "reflect", "random"])
        self.set_parameters(["epoch", "pop_size", "mut_rand", "patience", "restart_rate", "bound"])
        self.is_parallelizable = False
        self.sort_flag = False

    def initialize_variables(self):
        self._stall_count = 0
        self._stall_fit = None

    def before_main_loop(self):
        if self.g_best is not None and self.g_best.target is not None:
            self._stall_fit = self.g_best.target.fitness

    def restart_population(self):
        n_restart = int(round(self.pop_size * self.restart_rate))
        n_restart = np.clip(n_restart, 0, self.pop_size - 1)
        _, indices = self.get_sorted_population(self.pop, self.problem.minmax, return_index=True)
        worst_ids = indices[-n_restart:]
        new_agents = self.generate_population(n_restart)
        for idx, agent in zip(worst_ids, new_agents):
            self.pop[idx] = agent

    def restart_on_stagnation(self):
        if self.patience <= 0:
            return
        best = self.get_best_agent(self.pop, self.problem.minmax)

        if self._stall_fit is None or self.compare_fitness(best.target.fitness, self._stall_fit, self.problem.minmax):
            self._stall_fit = best.target.fitness
            self._stall_count = 0
            return

        self._stall_count += 1
        if self._stall_count >= self.patience:
            self._stall_count = 0
            self.restart_population()
            best = self.get_best_agent(self.pop, self.problem.minmax)
            self._stall_fit = best.target.fitness

    def reflect_solution(self, solution: np.ndarray) -> np.ndarray:
        lb = self.problem.lb
        ub = self.problem.ub

        x = solution.copy()
        span = ub - lb
        valid = span > 0
        x[~valid] = lb[~valid]
        if np.any(valid):
            offset = (x[valid] - lb[valid]) % (2.0 * span[valid])
            offset = np.where(offset > span[valid], 2.0 * span[valid] - offset, offset)
            x[valid] = lb[valid] + offset
        return x

    def random_solution(self, solution: np.ndarray) -> np.ndarray:
        x = solution.copy()
        mask = (x < self.problem.lb) | (x > self.problem.ub)
        if np.any(mask):
            x[mask] = self.generator.uniform(self.problem.lb[mask], self.problem.ub[mask])
        return x

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        if self.bound == "clip":
            return solution
        if self.bound == "random":
            return self.random_solution(solution)
        return self.reflect_solution(solution)

    def evolve(self, epoch):
        """
        Execute one iteration of WOAmM: modified mutualism phase followed by standard WOA moves.

        Args:
            epoch (int): The current iteration
        """
        # Modified mutualism phase
        base_pop = self.pop
        new_pop = [agent.copy() for agent in self.pop]
        for idx in range(0, self.pop_size):
            id_m, id_n = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), size=2, replace=False)
            # Paper: X_bf = best among Xi, Xm, Xn
            # X_other = worse of Xm, Xn (NOT Xi, since Xi is always updated via Eq.15)
            three_agents = [(idx, base_pop[idx].target),
                           (id_m, base_pop[id_m].target),
                           (id_n, base_pop[id_n].target)]
            # Find best among all three for X_bf
            if self.problem.minmax == "min":
                id_best = min(three_agents, key=lambda x: x[1].fitness)[0]
            else:
                id_best = max(three_agents, key=lambda x: x[1].fitness)[0]
            # X_other is the worse of Xm, Xn (never Xi)
            if self.compare_target(base_pop[id_m].target, base_pop[id_n].target, self.problem.minmax):
                id_other = id_n  # id_n is worse
            else:
                id_other = id_m  # id_m is worse
            pos_i = base_pop[idx].solution
            pos_other = base_pop[id_other].solution
            pos_best = base_pop[id_best].solution
            mutual_vector = (pos_i + pos_other) / 2
            bf1, bf2 = self.generator.integers(1, 3, 2)
            if self.mut_rand:
                rnd_i = self.generator.random(self.problem.n_dims)
                rnd_j = self.generator.random(self.problem.n_dims)
            else:
                rnd_i = self.generator.random()
                rnd_j = self.generator.random()
            xi_new = pos_i + rnd_i * (pos_best - mutual_vector * bf1)
            xj_new = pos_other + rnd_j * (pos_best - mutual_vector * bf2)
            xi_new = self.correct_solution(xi_new)
            xj_new = self.correct_solution(xj_new)
            xi_target = self.get_target(xi_new)
            xj_target = self.get_target(xj_new)
            if self.compare_target(xi_target, new_pop[idx].target, self.problem.minmax):
                new_pop[idx].update(solution=xi_new, target=xi_target)
            if self.compare_target(xj_target, new_pop[id_other].target, self.problem.minmax):
                new_pop[id_other].update(solution=xj_new, target=xj_target)
        self.pop = new_pop

        # Refresh global best before WOA movement (keep best-so-far)
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        self.g_best = self.get_better_agent(current_best, self.g_best, self.problem.minmax)

        # Standard WOA phase (like OriginalWOA with vector A, C per dimension)
        a = 2 - 2 * epoch / self.epoch      # linearly decreased from 2 to 0
        a2 = -1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            A = 2 * a * r1 - a
            C = 2 * r2
            bb = 1
            ll = (a2 - 1) * self.generator.random() + 1
            pp = self.generator.random(self.problem.n_dims)

            pos_new = self.pop[idx].solution.copy()
            for jdx in range(0, self.problem.n_dims):
                if pp[jdx] < 0.5:
                    if np.abs(A[jdx]) >= 1:
                        id_r = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                        D_X_rand = abs(C[jdx] * self.pop[id_r].solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.pop[id_r].solution[jdx] - A[jdx] * D_X_rand
                    else:
                        D_Leader = abs(C[jdx] * self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.g_best.solution[jdx] - A[jdx] * D_Leader
                else:
                    D1 = abs(self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                    pos_new[jdx] = D1 * np.exp(bb * ll) * np.cos(ll * 2 * np.pi) + self.g_best.solution[jdx]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].update(solution=pos_new, target=self.get_target(pos_new))
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)
        self.restart_on_stagnation()


class DevWOAmM(OriginalWOAmM):
    """
    Our developed version of: Whale Optimization Algorithm with Modified Mutualism (WOAmM)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Population size, in range [5, 10000]. Default is 100.
    mut_rand : bool
        Whether mutualism random coefficients are generated per dimension. Default is False.
    patience : int
        Number of stagnant epochs before restarting worst agents. Set 0 to disable, in range [0, 100000]. Default is 0.
    restart_rate : float
        Ratio of worst agents to restart when stagnation occurs, in range [0.0, 1.0]. Default is 0.2.
    bound : str
        Boundary handling method. Supported: "clip", "reflect", "random". Default is "clip".

    Note
    ----
    This version replaces the population after the WOA phase (no greedy selection).

    References
    ~~~~~~~~~~
    1. Chakraborty, S., Saha, A. K., Sharma, S., Mirjalili, S., & Chakraborty, R. (2021).
       A novel enhanced whale optimization algorithm for global optimization. Computers & Industrial Engineering, 153, 107086.
       https://doi.org/10.1016/j.cie.2020.107086
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, mut_rand: bool = False, patience: int = 0,
                 restart_rate: float = 0.2, bound: str = "clip", **kwargs) -> None:
        """
        Args:
            epoch: Maximum number of iterations.
            pop_size: Population size.
            mut_rand: Whether mutualism random coefficients are generated per dimension.
            patience: Number of stagnant epochs before restarting worst agents. Set 0 to disable.
            restart_rate: Ratio of worst agents to restart when stagnation occurs.
            bound: Boundary handling method. Supported: "clip", "reflect", "random".
        """
        super().__init__(epoch, pop_size, mut_rand, patience, restart_rate, bound, **kwargs)

    def evolve(self, epoch):
        """
        Execute one iteration of WOAmM: modified mutualism phase followed by standard WOA moves.

        Args:
            epoch (int): The current iteration
        """
        # Modified mutualism phase
        base_pop = self.pop
        new_pop = [agent.copy() for agent in self.pop]
        for idx in range(0, self.pop_size):
            id_m, id_n = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), size=2, replace=False)
            # Paper: X_bf = best among Xi, Xm, Xn
            # X_other = worse of Xm, Xn (NOT Xi, since Xi is always updated via Eq.15)
            three_agents = [(idx, base_pop[idx].target),
                           (id_m, base_pop[id_m].target),
                           (id_n, base_pop[id_n].target)]
            # Find best among all three for X_bf
            if self.problem.minmax == "min":
                id_best = min(three_agents, key=lambda x: x[1].fitness)[0]
            else:
                id_best = max(three_agents, key=lambda x: x[1].fitness)[0]
            # X_other is the worse of Xm, Xn (never Xi)
            if self.compare_target(base_pop[id_m].target, base_pop[id_n].target, self.problem.minmax):
                id_other = id_n  # id_n is worse
            else:
                id_other = id_m  # id_m is worse
            pos_i = base_pop[idx].solution
            pos_other = base_pop[id_other].solution
            pos_best = base_pop[id_best].solution
            mutual_vector = (pos_i + pos_other) / 2
            bf1, bf2 = self.generator.integers(1, 3, 2)
            if self.mut_rand:
                rnd_i = self.generator.random(self.problem.n_dims)
                rnd_j = self.generator.random(self.problem.n_dims)
            else:
                rnd_i = self.generator.random()
                rnd_j = self.generator.random()
            xi_new = pos_i + rnd_i * (pos_best - mutual_vector * bf1)
            xj_new = pos_other + rnd_j * (pos_best - mutual_vector * bf2)
            xi_new = self.correct_solution(xi_new)
            xj_new = self.correct_solution(xj_new)
            xi_target = self.get_target(xi_new)
            xj_target = self.get_target(xj_new)
            if self.compare_target(xi_target, new_pop[idx].target, self.problem.minmax):
                new_pop[idx].update(solution=xi_new, target=xi_target)
            if self.compare_target(xj_target, new_pop[id_other].target, self.problem.minmax):
                new_pop[id_other].update(solution=xj_new, target=xj_target)
        self.pop = new_pop

        # Refresh global best before WOA movement (keep best-so-far)
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        self.g_best = self.get_better_agent(current_best, self.g_best, self.problem.minmax)

        # Standard WOA phase (like OriginalWOA with vector A, C per dimension)
        a = 2 - 2 * epoch / self.epoch      # linearly decreased from 2 to 0
        a2 = -1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            A = 2 * a * r1 - a
            C = 2 * r2
            bb = 1
            ll = (a2 - 1) * self.generator.random() + 1
            pp = self.generator.random(self.problem.n_dims)

            pos_new = self.pop[idx].solution.copy()
            for jdx in range(0, self.problem.n_dims):
                if pp[jdx] < 0.5:
                    if np.abs(A[jdx]) >= 1:
                        id_r = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                        D_X_rand = abs(C[jdx] * self.pop[id_r].solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.pop[id_r].solution[jdx] - A[jdx] * D_X_rand
                    else:
                        D_Leader = abs(C[jdx] * self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.g_best.solution[jdx] - A[jdx] * D_Leader
                else:
                    D1 = abs(self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                    pos_new[jdx] = D1 * np.exp(bb * ll) * np.cos(ll * 2 * np.pi) + self.g_best.solution[jdx]

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        self.restart_on_stagnation()
