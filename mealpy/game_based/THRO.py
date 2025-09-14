#!/usr/bin/env python
# Created by "Thieu" at 21:42, 13/09/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalTHRO(Optimizer):
    """
    The original version of: Tianji's Horse Racing Optimization (THRO)

    Links:
        + https://www.mathworks.com/matlabcentral/fileexchange/181341-tianji-s-horse-racing-optimization-thro

    Notes:
        + This algorithm has many drawbacks, especially in training, where such cases almost never occur.
        As a result, scenarios 3, 4, and 5 will practically never happen, since the situation where
        two solutions have exactly the same fitness value is very rare in practice.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, THRO
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
    >>> model = THRO.OriginalTHRO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, L., Du, H., Zhang, Z., Hu, G., Mirjalili, S., Khodadadi, N., Hussien, A.G., Liao, Y. and Zhao, W., 2025.
    Tianji’s horse racing optimization (THRO): a new metaheuristic inspired by ancient wisdom and its
    engineering optimization applications. Artificial Intelligence Review, 58(9), p.282.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.is_parallelizable = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.clip(solution, self.problem.lb, self.problem.ub)

    def before_main_loop(self):
        # Split to two groups: tianji and king (50%-50%)
        self.n_pop = self.pop_size // 2
        self.pop_tianji = self.pop[:self.n_pop]
        self.pop_king = self.pop[self.n_pop:]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        
        ### """Main racing phase with five scenarios"""
        # Randomly shuffle and redistribute populations
        self.pop = self.rng.sample(self.pop, len(self.pop))
        self.pop_tianji = self.pop[:self.n_pop].copy()
        self.pop_king = self.pop[self.n_pop:].copy()
        
        # Sort populations by fitness
        self.pop_tianji = self.get_sorted_population(self.pop_tianji, self.problem.minmax)
        self.pop_king = self.get_sorted_population(self.pop_king, self.problem.minmax)

        # Generate binary matrices T_B and K_B
        t_b = np.zeros((self.n_pop, self.problem.n_dims))
        k_b = np.zeros((self.n_pop, self.problem.n_dims))

        for idx in range(self.n_pop):
            # For Tianji
            rand_dim = self.generator.permutation(self.problem.n_dims)
            rand_num = int(np.ceil(np.sin(np.pi / 2 * self.generator.random()) * self.problem.n_dims))
            t_b[idx, rand_dim[:rand_num]] = 1

            # For King
            rand_dim = self.generator.permutation(self.problem.n_dims)
            rand_num = int(np.ceil(np.sin(np.pi / 2 * self.generator.random()) * self.problem.n_dims))
            k_b[idx, rand_dim[:rand_num]] = 1
            
        # Weight parameter
        p = 1 - epoch / self.epoch

        # Current horse indices
        tianji_slowest_id = self.n_pop - 1
        tianji_fastest_id = 0
        king_slowest_id = self.n_pop - 1
        king_fastest_id = 0

        # Racing scenarios
        for idx in range(self.n_pop):
            tianji_alpha = 1 + np.round(0.5 * (0.5 + self.generator.random())) * self.generator.standard_normal()
            t_beta = np.round(0.5 * (0.1 + self.generator.random())) * self.generator.standard_normal()
            king_alpha = 1 + np.round(0.5 * (0.5 + self.generator.random())) * self.generator.standard_normal()
            k_beta = np.round(0.5 * (0.1 + self.generator.random())) * self.generator.standard_normal()

            tianji_r = self.get_levy_flight_step(beta=1.5, multiplier=1, size=None, case=-1) * t_b[idx]
            king_r = self.get_levy_flight_step(beta=1.5, multiplier=1, size=None, case=-1) * k_b[idx]

            fit_t = self.pop_tianji[tianji_slowest_id].target.fitness
            fit_k = self.pop_king[king_slowest_id].target.fitness
            if self.problem.minmax == "min":
                if fit_t < fit_k:
                    case = 1
                elif fit_t > fit_k:
                    case = 2
                else:
                    case = 3
            else:
                if fit_t > fit_k:
                    case = 1
                elif fit_t < fit_k:
                    case = 2
                else:
                    case = 3

            tianji_mean = np.mean([agent.solution for agent in self.pop_tianji], axis=0)
            king_mean = np.mean([agent.solution for agent in self.pop_king], axis=0)

            # Scenario 1: Tianji's slowest < King's slowest
            if case == 1:
                # Update Tianji's slowest horse
                pos_new = ((p * self.pop_tianji[tianji_slowest_id].solution + (1 - p) * self.pop_tianji[0].solution) +
                                  tianji_r * (self.pop_tianji[0].solution - self.pop_tianji[tianji_slowest_id].solution +
                                         p * (tianji_mean - king_mean))) * tianji_alpha + t_beta
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, self.pop_tianji[tianji_slowest_id].target, self.problem.minmax):
                    self.pop_tianji[tianji_slowest_id] = agent

                # Update King's slowest horse
                pos_new = ((p * self.pop_king[king_slowest_id].solution + (1 - p) * self.pop_tianji[tianji_slowest_id].solution) +
                                king_r * (self.pop_tianji[tianji_slowest_id].solution - self.pop_king[king_slowest_id].solution +
                                       p * (tianji_mean - king_mean))) * king_alpha + k_beta
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, self.pop_king[king_slowest_id].target, self.problem.minmax):
                    self.pop_king[king_slowest_id] = agent

                tianji_slowest_id = max(0, tianji_slowest_id - 1)
                king_slowest_id = max(0, king_slowest_id - 1)

            # Scenario 2: Tianji's slowest > King's slowest
            elif case == 2:
                tr1 = self.generator.choice(list(set(range(self.n_pop)) - {tianji_slowest_id}))
                # Update Tianji's slowest horse
                pos_new = ((p * self.pop_tianji[tianji_slowest_id].solution + (1 - p) * self.pop_tianji[tr1].solution) +
                                  tianji_r * (self.pop_tianji[tr1].solution - self.pop_tianji[tianji_slowest_id].solution +
                                         p * (tianji_mean - king_mean))) * tianji_alpha + t_beta
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, self.pop_tianji[tianji_slowest_id].target, self.problem.minmax):
                    self.pop_tianji[tianji_slowest_id] = agent

                # Update King's fastest horse
                pos_new = ((p * self.pop_king[king_fastest_id].solution + (1 - p) * self.pop_king[0].solution) +
                                king_r * (self.pop_king[0].solution - self.pop_king[king_fastest_id].solution +
                                       p * (tianji_mean - king_mean))) * king_alpha + k_beta
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, self.pop_king[king_fastest_id].target, self.problem.minmax):
                    self.pop_king[king_fastest_id] = agent

                tianji_slowest_id = max(0, tianji_slowest_id - 1)
                king_fastest_id = min(self.n_pop - 1, king_fastest_id + 1)

            else:  # Equal slowest speeds
                fit_t = self.pop_tianji[tianji_fastest_id].target.fitness
                fit_k = self.pop_king[king_fastest_id].target.fitness
                if self.problem.minmax == "min":
                    if fit_t < fit_k:
                        case_fast = 1
                    elif fit_t > fit_k:
                        case_fast = 2
                    else:
                        case_fast = 3
                else:
                    if fit_t > fit_k:
                        case_fast = 1
                    elif fit_t < fit_k:
                        case_fast = 2
                    else:
                        case_fast = 3

                # Scenario 3: Tianji's fastest < King's fastest
                if case_fast == 1:
                    # Update Tianji's fastest horse
                    pos_new = ((p * self.pop_tianji[tianji_fastest_id].solution + (1 - p) * self.pop_tianji[0].solution) +
                                      tianji_r * (self.pop_tianji[0].solution - self.pop_tianji[tianji_fastest_id].solution +
                                             p * (tianji_mean - king_mean))) * tianji_alpha + t_beta
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(agent.target, self.pop_tianji[tianji_fastest_id].target, self.problem.minmax):
                        self.pop_tianji[tianji_fastest_id] = agent

                    # Update King's fastest horse
                    pos_new = ((p * self.pop_king[king_fastest_id].solution + (1 - p) * self.pop_tianji[tianji_fastest_id].solution) +
                                    king_r * (self.pop_tianji[tianji_fastest_id].solution - self.pop_king[king_fastest_id].solution +
                                           p * (tianji_mean - king_mean))) * king_alpha + k_beta
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(agent.target, self.pop_king[king_fastest_id].target, self.problem.minmax):
                        self.pop_king[king_fastest_id] = agent

                    tianji_fastest_id = min(self.n_pop - 1, tianji_fastest_id + 1)
                    king_fastest_id = min(self.n_pop - 1, king_fastest_id + 1)

                # Scenario 4: Tianji's fastest > King's fastest
                elif case_fast == 2:
                    tr2 = self.generator.choice(list(set(range(self.n_pop)) - {tianji_fastest_id}))

                    # Update Tianji's slowest horse
                    pos_new = ((p * self.pop_tianji[tianji_slowest_id].solution + (1 - p) * self.pop_tianji[tr2].solution) +
                                      tianji_r * (self.pop_tianji[tr2].solution - self.pop_tianji[tianji_slowest_id].solution +
                                             p * (tianji_mean - king_mean))) * tianji_alpha + t_beta
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(agent.target, self.pop_tianji[tianji_slowest_id].target, self.problem.minmax):
                        self.pop_tianji[tianji_slowest_id] = agent

                    # Update King's fastest horse
                    pos_new = ((p * self.pop_king[king_fastest_id].solution + (1 - p) * self.pop_king[0].solution) +
                                    king_r * (self.pop_king[0].solution - self.pop_king[king_fastest_id].solution +
                                           p * (tianji_mean - king_mean))) * king_alpha + k_beta
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(agent.target, self.pop_king[king_fastest_id].target, self.problem.minmax):
                        self.pop_king[king_fastest_id] = agent

                    tianji_slowest_id = max(0, tianji_slowest_id - 1)
                    king_fastest_id = min(self.n_pop - 1, king_fastest_id + 1)

                # Scenario 5: Equal fastest speeds
                else:
                    tr3 = self.generator.choice(list(set(range(self.n_pop)) - {tianji_slowest_id}))

                    # Update Tianji's slowest horse
                    pos_new = ((p * self.pop_tianji[tianji_slowest_id].solution + (1 - p) * self.pop_tianji[tr3].solution) +
                                      tianji_r * (self.pop_tianji[tr3].solution - self.pop_tianji[tianji_slowest_id].solution +
                                             p * (tianji_mean - king_mean))) * tianji_alpha + t_beta
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(agent.target, self.pop_tianji[tianji_slowest_id].target, self.problem.minmax):
                        self.pop_tianji[tianji_slowest_id] = agent

                    # Update King's fastest horse
                    pos_new = ((p * self.pop_king[king_fastest_id].solution + (1 - p) * self.pop_king[0].solution) +
                                    king_r * (self.pop_king[0].solution - self.pop_king[king_fastest_id].solution +
                                           p * (tianji_mean - king_mean))) * king_alpha + k_beta
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(agent.target, self.pop_king[king_fastest_id].target, self.problem.minmax):
                        self.pop_king[king_fastest_id] = agent

                    tianji_slowest_id = max(0, tianji_slowest_id - 1)
                    king_fastest_id = min(self.n_pop - 1, king_fastest_id + 1)

        ## Update global best
        self.update_global_best_agent(self.pop_tianji + self.pop_king, save=False)

        # Training phase
        best_tianji = self.get_best_agent(self.pop_tianji, self.problem.minmax)
        best_king = self.get_best_agent(self.pop_king, self.problem.minmax)

        for idx in range(self.n_pop):
            # Training for Tianji's population
            pos_new = self.pop_tianji[idx].solution
            for jdx in range(self.problem.n_dims):
                if self.generator.random() > 0.5:       # Levy flight based training
                    tr4, tr5 = self.generator.choice(list(set(range(self.n_pop)) - {idx}), size=2, replace=False)
                    lt = self.get_levy_flight_step(beta=1.5, multiplier=0.2, size=None, case=-1)
                    pos_new[jdx] = pos_new[jdx] + lt * (self.pop_tianji[tr4].solution[jdx] - self.pop_tianji[tr5].solution[jdx])
                else:        # Best-guided training
                    mt = 0.5 * (1 + 0.001 * (1 - epoch / self.epoch) ** 2 * np.sin(np.pi * self.generator.random()))
                    pos_new[jdx] = best_tianji.solution[jdx] + mt * (best_tianji.solution[jdx] - pos_new[jdx])
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop_tianji[idx].target, self.problem.minmax):
                self.pop_tianji[idx] = agent

            # Training for King's population
            pos_new = self.pop_tianji[idx].solution
            for jdx in range(self.problem.n_dims):
                if self.generator.random() > 0.5:       # Levy flight based training
                    kr1, kr2 = self.generator.choice(list(set(range(self.n_pop)) - {idx}), size=2, replace=False)
                    lk = self.get_levy_flight_step(beta=1.5, multiplier=0.2, size=None, case=-1)
                    pos_new[jdx] = pos_new[jdx] + lk * (self.pop_king[kr1].solution[jdx] - self.pop_king[kr2].solution[jdx])
                else:       # Best-guided training
                    mk = 0.5 * (1 + 0.001 * (1 - epoch / self.epoch) ** 2 * np.sin(np.pi * self.generator.random()))
                    pos_new[jdx] = best_king.solution[jdx] + mk * (best_king.solution[jdx] - pos_new[jdx])
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop_king[idx].target, self.problem.minmax):
                self.pop_king[idx] = agent

        # Merge populations back
        self.pop = self.pop_tianji + self.pop_king
