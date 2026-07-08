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

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2016.01.008
        2. https://mathworks.com/matlabcentral/fileexchange/55667-the-whale-optimization-algorithm

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

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.
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
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        if self.epoch > 1:
            progress = (epoch - 1) / (self.epoch - 1)
        else:
            progress = 1.0
        a = 2 - 2 * progress  # linearly decreased from 2 to 0
        a2 = -1 - progress
        pop_new = []
        for idx in range(0, self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = (a2 - 1) * self.generator.random() + 1
            p = self.generator.random()

            pos_new = self.pop[idx].solution.copy()
            for jdx in range(0, self.problem.n_dims):
                if p < 0.5:
                    if np.abs(A[jdx]) >= 1:
                        id_r = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                        D_X_rand = abs(C[jdx] * self.pop[id_r].solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.pop[id_r].solution[jdx] - A[jdx] * D_X_rand
                    else:
                        D_Leader = abs(C[jdx] * self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.g_best.solution[jdx] - A[jdx] * D_Leader
                else:
                    D1 = abs(self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                    pos_new[jdx] = D1 * np.exp(b * l) * np.cos(l * 2 * np.pi) + self.g_best.solution[jdx]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].update(solution=pos_new, target=self.get_target(pos_new))
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)


class DevWOA(Optimizer):
    """
    The developed version of: Whale Optimization Algorithm (WOA)

    Notes:
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

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.
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
            r = self.generator.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.generator.uniform(-1, 1)
            p = 0.5
            b = 1

            # Get pos1
            pos1 = self.g_best.solution - A * np.abs(C * self.g_best.solution - self.pop[idx].solution)

            # Get pos2
            id_r2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            pos2 = self.pop[id_r2].solution - A * np.abs(C * self.pop[id_r2].solution - self.pop[idx].solution)

            # Get pos3
            D1 = np.abs(self.g_best.solution - self.pop[idx].solution)
            pos3 = self.g_best.solution + np.exp(b * l) * np.cos(2 * np.pi * l) * D1

            # Get final pos_new
            pos_new = pos1 if np.abs(A) < 1 else pos2
            pos_new = np.where(self.generator.random(size=self.problem.n_dims) < p, pos_new, pos3)

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

    Links:
        1. https://ieenp.explore.ieee.org/document/8900003

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + feedback_max (int): maximum iterations of each feedback, default = 10

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

    References
    ~~~~~~~~~~
    [1] Tang, C., Sun, W., Wu, W. and Xue, M., 2019, July. A hybrid improved whale optimization algorithm.
    In 2019 IEEE 15th International Conference on Control and Automation (ICCA) (pp. 362-367). IEEE.
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
            r = self.generator.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.generator.uniform(-1, 1)
            p = 0.5
            b = 1
            if self.generator.uniform() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best.solution - self.pop[idx].solution)
                    pos_new = self.g_best.solution - A * D
                else:
                    # x_rand = pop[self.generator.self.generator.randint(self.pop_size)]         # select random 1 position in pop
                    x_rand = self.problem.generate_solution()
                    D = np.abs(C * x_rand - self.pop[idx].solution)
                    pos_new = x_rand - A * D
            else:
                D1 = np.abs(self.g_best.solution - self.pop[idx].solution)
                pos_new = self.g_best.solution + np.exp(b * l) * np.cos(2 * np.pi * l) * D1
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

class WOAmM(Optimizer):
    """
    The WOAmM metaheuristic: Whale Optimization Algorithm with Modified Mutualism phase

    References
    ~~~~~~~~~~
    [1] Chakraborty, S., Saha, A. K., Sharma, S., Mirjalili, S., & Chakraborty, R. (2021).
    A novel enhanced whale optimization algorithm for global optimization. Computers & Industrial Engineering, 153, 107086.
    https://doi.org/10.1016/j.cie.2020.107086
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        mutualism_vector_rand: bool = False,
        mutualism_snapshot: bool = False,
        stagnation_epochs: int = 0,
        restart_ratio: float = 0.2,
        boundary_mode: str = "clip",
        **kwargs: object,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mutualism_vector_rand = self.validator.check_bool("mutualism_vector_rand", mutualism_vector_rand)
        self.mutualism_snapshot = self.validator.check_bool("mutualism_snapshot", mutualism_snapshot)
        self.stagnation_epochs = self.validator.check_int("stagnation_epochs", stagnation_epochs, [0, 100000])
        self.restart_ratio = self.validator.check_float("restart_ratio", restart_ratio, [0.0, 1.0])
        self.boundary_mode = self.validator.check_str("boundary_mode", boundary_mode, ["clip", "reflect", "random"])
        self.set_parameters([
            "epoch",
            "pop_size",
            "mutualism_vector_rand",
            "mutualism_snapshot",
            "stagnation_epochs",
            "restart_ratio",
            "boundary_mode",
        ])
        self.is_parallelizable = False
        self.sort_flag = False

    def initialize_variables(self):
        self._stall_count = 0
        self._stall_best_fitness = None

    def before_main_loop(self):
        if self.g_best is not None and self.g_best.target is not None:
            self._stall_best_fitness = self.g_best.target.fitness

    def _restart_population(self):
        ratio = min(max(self.restart_ratio, 0.0), 1.0)
        if ratio <= 0:
            return
        n_restart = int(round(self.pop_size * ratio))
        n_restart = min(max(n_restart, 0), self.pop_size - 1)
        if n_restart <= 0:
            return
        _, indices = self.get_sorted_population(self.pop, self.problem.minmax, return_index=True)
        worst_indices = indices[-n_restart:]
        new_agents = self.generate_population(n_restart)
        for idx, agent in zip(worst_indices, new_agents):
            self.pop[idx] = agent

    def _maybe_restart_on_stagnation(self):
        if self.stagnation_epochs <= 0:
            return
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        if self._stall_best_fitness is None or self.compare_fitness(
            current_best.target.fitness, self._stall_best_fitness, self.problem.minmax
        ):
            self._stall_best_fitness = current_best.target.fitness
            self._stall_count = 0
            return
        self._stall_count += 1
        if self._stall_count >= self.stagnation_epochs:
            self._stall_count = 0
            self._restart_population()
            current_best = self.get_best_agent(self.pop, self.problem.minmax)
            self._stall_best_fitness = current_best.target.fitness

    def _reflect_solution(self, solution: np.ndarray) -> np.ndarray:
        lb = self.problem.lb
        ub = self.problem.ub
        x = solution.copy()
        for idx in range(len(x)):
            span = ub[idx] - lb[idx]
            if span <= 0:
                x[idx] = lb[idx]
                continue
            val = x[idx]
            if val < lb[idx] or val > ub[idx]:
                offset = (val - lb[idx]) % (2 * span)
                if offset > span:
                    offset = 2 * span - offset
                x[idx] = lb[idx] + offset
        return x

    def _random_solution(self, solution: np.ndarray) -> np.ndarray:
        lb = self.problem.lb
        ub = self.problem.ub
        x = solution.copy()
        mask = (x < lb) | (x > ub)
        if np.any(mask):
            x[mask] = self.generator.uniform(lb[mask], ub[mask])
        return x

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        if self.boundary_mode == "clip":
            return solution
        if self.boundary_mode == "random":
            return self._random_solution(solution)
        return self._reflect_solution(solution)

    def evolve(self, epoch):
        """
        Execute one iteration of WOAmM: modified mutualism phase followed by standard WOA moves.

        Args:
            epoch (int): The current iteration
        """
        # Modified mutualism phase
        base_pop = self.pop
        new_pop = self.pop
        if self.mutualism_snapshot:
            base_pop = [agent.copy() for agent in self.pop]
            new_pop = [agent.copy() for agent in self.pop]
        for idx in range(0, self.pop_size):
            id_candidates = list(set(range(0, self.pop_size)) - {idx})
            id_m, id_n = self.generator.choice(id_candidates, size=2, replace=False)
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
            if self.mutualism_vector_rand:
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
        if self.mutualism_snapshot:
            self.pop = new_pop

        # Refresh global best before WOA movement (keep best-so-far)
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        self.g_best = self.get_better_agent(current_best, self.g_best, self.problem.minmax)

        # Standard WOA phase (like OriginalWOA with vector A, C per dimension)
        if self.epoch > 1:
            progress = (epoch - 1) / (self.epoch - 1)
        else:
            progress = 1.0
        a = 2 - 2 * progress
        a2 = -1 - progress
        b = 1
        pop_new = []
        for idx in range(0, self.pop_size):
            # Use vector A, C (per dimension) like OriginalWOA
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            A = 2 * a * r1 - a
            C = 2 * r2
            l = (a2 - 1) * self.generator.random() + 1
            p = self.generator.random()

            pos_new = self.pop[idx].solution.copy()
            for jdx in range(0, self.problem.n_dims):
                if p < 0.5:
                    if np.abs(A[jdx]) >= 1:
                        id_r = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                        D_X_rand = abs(C[jdx] * self.pop[id_r].solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.pop[id_r].solution[jdx] - A[jdx] * D_X_rand
                    else:
                        D_Leader = abs(C[jdx] * self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.g_best.solution[jdx] - A[jdx] * D_Leader
                else:
                    D1 = abs(self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                    pos_new[jdx] = D1 * np.exp(b * l) * np.cos(l * 2 * np.pi) + self.g_best.solution[jdx]

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].update(solution=pos_new, target=self.get_target(pos_new))
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)
        self._maybe_restart_on_stagnation()


class WOAmM_Paper(Optimizer):
    """
    WOAmM variant that follows the paper's population replacement behavior.
    This version replaces the population after the WOA phase (no greedy selection).
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        mutualism_vector_rand: bool = False,
        mutualism_snapshot: bool = False,
        stagnation_epochs: int = 0,
        restart_ratio: float = 0.2,
        boundary_mode: str = "clip",
        **kwargs: object,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mutualism_vector_rand = self.validator.check_bool("mutualism_vector_rand", mutualism_vector_rand)
        self.mutualism_snapshot = self.validator.check_bool("mutualism_snapshot", mutualism_snapshot)
        self.stagnation_epochs = self.validator.check_int("stagnation_epochs", stagnation_epochs, [0, 100000])
        self.restart_ratio = self.validator.check_float("restart_ratio", restart_ratio, [0.0, 1.0])
        self.boundary_mode = self.validator.check_str("boundary_mode", boundary_mode, ["clip", "reflect", "random"])
        self.set_parameters([
            "epoch",
            "pop_size",
            "mutualism_vector_rand",
            "mutualism_snapshot",
            "stagnation_epochs",
            "restart_ratio",
            "boundary_mode",
        ])
        self.is_parallelizable = False
        self.sort_flag = False

    def initialize_variables(self):
        self._stall_count = 0
        self._stall_best_fitness = None

    def before_main_loop(self):
        if self.g_best is not None and self.g_best.target is not None:
            self._stall_best_fitness = self.g_best.target.fitness

    def _restart_population(self):
        ratio = min(max(self.restart_ratio, 0.0), 1.0)
        if ratio <= 0:
            return
        n_restart = int(round(self.pop_size * ratio))
        n_restart = min(max(n_restart, 0), self.pop_size - 1)
        if n_restart <= 0:
            return
        _, indices = self.get_sorted_population(self.pop, self.problem.minmax, return_index=True)
        worst_indices = indices[-n_restart:]
        new_agents = self.generate_population(n_restart)
        for idx, agent in zip(worst_indices, new_agents):
            self.pop[idx] = agent

    def _maybe_restart_on_stagnation(self):
        if self.stagnation_epochs <= 0:
            return
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        if self._stall_best_fitness is None or self.compare_fitness(
            current_best.target.fitness, self._stall_best_fitness, self.problem.minmax
        ):
            self._stall_best_fitness = current_best.target.fitness
            self._stall_count = 0
            return
        self._stall_count += 1
        if self._stall_count >= self.stagnation_epochs:
            self._stall_count = 0
            self._restart_population()
            current_best = self.get_best_agent(self.pop, self.problem.minmax)
            self._stall_best_fitness = current_best.target.fitness

    def _reflect_solution(self, solution: np.ndarray) -> np.ndarray:
        lb = self.problem.lb
        ub = self.problem.ub
        x = solution.copy()
        for idx in range(len(x)):
            span = ub[idx] - lb[idx]
            if span <= 0:
                x[idx] = lb[idx]
                continue
            val = x[idx]
            if val < lb[idx] or val > ub[idx]:
                offset = (val - lb[idx]) % (2 * span)
                if offset > span:
                    offset = 2 * span - offset
                x[idx] = lb[idx] + offset
        return x

    def _random_solution(self, solution: np.ndarray) -> np.ndarray:
        lb = self.problem.lb
        ub = self.problem.ub
        x = solution.copy()
        mask = (x < lb) | (x > ub)
        if np.any(mask):
            x[mask] = self.generator.uniform(lb[mask], ub[mask])
        return x

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        if self.boundary_mode == "clip":
            return solution
        if self.boundary_mode == "random":
            return self._random_solution(solution)
        return self._reflect_solution(solution)

    def evolve(self, epoch):
        """
        Execute one iteration of WOAmM (paper mode): mutualism phase then WOA update.

        Args:
            epoch (int): The current iteration
        """
        # Modified mutualism phase
        base_pop = self.pop
        new_pop = self.pop
        if self.mutualism_snapshot:
            base_pop = [agent.copy() for agent in self.pop]
            new_pop = [agent.copy() for agent in self.pop]
        for idx in range(0, self.pop_size):
            id_candidates = list(set(range(0, self.pop_size)) - {idx})
            id_m, id_n = self.generator.choice(id_candidates, size=2, replace=False)
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
            if self.mutualism_vector_rand:
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
        if self.mutualism_snapshot:
            self.pop = new_pop

        # Refresh global best before WOA movement (keep best-so-far)
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        self.g_best = self.get_better_agent(current_best, self.g_best, self.problem.minmax)

        # Standard WOA phase (like OriginalWOA with vector A, C per dimension)
        if self.epoch > 1:
            progress = (epoch - 1) / (self.epoch - 1)
        else:
            progress = 1.0
        a = 2 - 2 * progress
        a2 = -1 - progress
        b = 1
        pop_new = []
        for idx in range(0, self.pop_size):
            # Use vector A, C (per dimension) like OriginalWOA
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            A = 2 * a * r1 - a
            C = 2 * r2
            l = (a2 - 1) * self.generator.random() + 1
            p = self.generator.random()

            pos_new = self.pop[idx].solution.copy()
            for jdx in range(0, self.problem.n_dims):
                if p < 0.5:
                    if np.abs(A[jdx]) >= 1:
                        id_r = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                        D_X_rand = abs(C[jdx] * self.pop[id_r].solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.pop[id_r].solution[jdx] - A[jdx] * D_X_rand
                    else:
                        D_Leader = abs(C[jdx] * self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                        pos_new[jdx] = self.g_best.solution[jdx] - A[jdx] * D_Leader
                else:
                    D1 = abs(self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                    pos_new[jdx] = D1 * np.exp(b * l) * np.cos(l * 2 * np.pi) + self.g_best.solution[jdx]

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].update(solution=pos_new, target=self.get_target(pos_new))
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)
        self._maybe_restart_on_stagnation()
