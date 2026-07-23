#!/usr/bin/env python
# Created by "Thieu" at 17:18, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalPO(Optimizer):
    """
    The original version of: Political Optimizer (PO) Algorithm

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [2, 100]. Default is 8 (Please read the note below about this parameter).
    lamda_max : float
        Upper limit of the party switching rate, in range [1.0, 100.0]. Default is 1.0.


    .. attention::
       - `pop_size`: In this algorithm, the `pop_size` parameter corresponds to 'n' from the paper.
         It defines the number of political  parties and the number of electoral constituencies.
       - Actual Population Size: The true number of candidate solutions generated and evaluated
         is `pop_size ** 2`. For example, setting `pop_size = 8` (the paper's recommended value)
         yields an actual working population of 64 candidates (8 parties * 8 candidates).

    References
    ----------
    1. Askari, Q., Younas, I., & Saeed, M. (2020). Political Optimizer: A novel socio-inspired meta-heuristic
       for global optimization. Knowledge-based systems, 195, 105709. https://doi.org/10.1016/j.knosys.2020.105709

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, PO
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
    >>> model = PO.OriginalPO(epoch=1000, pop_size=10, lamda_max=1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 8, lamda_max: float=1.0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 8
            lamda_max (float) : Upper limit of the party switching rate, default=1.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_n = self.validator.check_int("pop_size", pop_size, [2, 100])
        self.pop_size = self.pop_n ** 2
        self.lamda_max = self.validator.check_int("lamda_max", lamda_max, [1.0, 100])
        self.set_parameters(["epoch", "pop_size", "lamda_max"])
        self.sort_flag = False
        self.lam = None
        self.pop_prev = None
        self.leaders_idx, self.winners_idx = None, None

    def before_main_loop(self):
        self.lam = self.lamda_max
        self.pop_prev = self.pop.copy()
        fitness = np.array([agent.target.fitness for agent in self.pop])
        self.leaders_idx, self.winners_idx = self.get_leaders_and_winners(fitness)

    def rppus_update(self, p_curr, p_prev, m_star, is_improved):
        """
        Recent Past-based Position Updating Strategy (RPPUS).
        """
        r = self.generator.random(self.problem.n_dims)

        # Determine the relationships between previous, current, and referenced positions
        c1 = ((p_prev <= p_curr) & (p_curr <= m_star)) | ((p_prev >= p_curr) & (p_curr >= m_star))
        c2 = ((p_prev <= m_star) & (m_star <= p_curr)) | ((p_prev >= m_star) & (m_star >= p_curr))
        c3 = ((m_star <= p_prev) & (p_prev <= p_curr)) | ((m_star >= p_prev) & (p_prev >= p_curr))

        # Eq (9) Updates for improved fitness
        eq9_u1 = m_star + r * (m_star - p_curr)
        eq9_u2 = m_star + (2 * r - 1) * np.abs(m_star - p_curr)
        eq9_u3 = m_star + (2 * r - 1) * np.abs(m_star - p_prev)

        # Eq (10) Updates for deteriorated fitness
        eq10_u1 = m_star + (2 * r - 1) * np.abs(m_star - p_curr)
        eq10_u2 = p_prev + r * (p_curr - p_prev)
        eq10_u3 = m_star + (2 * r - 1) * np.abs(m_star - p_prev)

        update = np.zeros_like(p_curr)
        # Apply Eq (9)
        if is_improved:
            update = np.where(c1, eq9_u1, update)
            update = np.where(c2, eq9_u2, update)
            update = np.where(c3, eq9_u3, update)
        # Apply Eq (10)
        else:
            update = np.where(c1, eq10_u1, update)
            update = np.where(c2, eq10_u2, update)
            update = np.where(c3, eq10_u3, update)

        # Fallback for any unassigned edge conditions
        unassigned = ~(c1 | c2 | c3)
        update = np.where(unassigned, m_star + r * (m_star - p_curr), update)
        return update

    def get_leaders_and_winners(self, fitness):
        """
        Extract the indices of the party leaders and constituency winners
        from the flattened 1D fitness array.
        """
        leaders_idx = np.zeros(self.pop_n, dtype=int)
        winners_idx = np.zeros(self.pop_n, dtype=int)
        # Each party i occupies indices from (i * n) to (i * n + n - 1)
        for idx in range(self.pop_n):
            start_idx = idx * self.pop_n
            end_idx = start_idx + self.pop_n
            if self.problem.minmax == "min":
                leaders_idx[idx] = start_idx + np.argmin(fitness[start_idx:end_idx])
            else:
                leaders_idx[idx] = start_idx + np.argmax(fitness[start_idx:end_idx])
        # Each constituency j occupies indices j, j+n, j+2n, ..., j+(n-1)n
        for jdx in range(self.pop_n):
            constituency_indices = np.arange(jdx, self.pop_size, self.pop_n)
            if self.problem.minmax == "min":
                winners_idx[jdx] = constituency_indices[np.argmin(fitness[constituency_indices])]
            else:
                winners_idx[jdx] = constituency_indices[np.argmax(fitness[constituency_indices])]
        return leaders_idx, winners_idx

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_temp = self.pop.copy()
        fitness = np.array([agent.target.fitness for agent in self.pop])

        # Election Campaign Phase
        for k in range(self.pop_size):
            party_idx = k // self.pop_n
            const_idx = k % self.pop_n

            is_improved = False
            if self.compare_target(self.pop[k].target, self.pop_prev[k].target, self.problem.minmax):
                is_improved = True

            # Update w.r.t the party leader
            m_party = self.pop[self.leaders_idx[party_idx]].solution
            pos_new = self.rppus_update(self.pop[k].solution, self.pop_prev[k].solution, m_party, is_improved)

            # Update w.r.t the constituency winner
            m_const = self.pop[self.winners_idx[const_idx]].solution
            pos_new = self.rppus_update(pos_new, self.pop_prev[k].solution, m_const, is_improved)

            # Apply boundary constraints
            self.pop[k].solution = self.amend_solution(pos_new)

        # Party Switching Phase
        for k in range(self.pop_size):
            if self.generator.random() < self.lam:
                r = self.generator.integers(self.pop_n)  # Select a random party

                # Find the least fit member of party r
                start_idx = r * self.pop_n
                end_idx = start_idx + self.pop_n
                if self.problem.minmax == "min":
                    q_local = np.argmax(fitness[start_idx:end_idx])
                else:
                    q_local = np.argmin(fitness[start_idx:end_idx])
                q = start_idx + q_local

                # Swap the candidate solutions
                agent = self.pop[q].copy()
                self.pop[q] = self.pop[k].copy()
                self.pop[k] = agent.copy()
                fitness[k], fitness[q] = fitness[q], fitness[k]

        # Election Phase: Recalculate fitness
        for idx in range(self.pop_size):
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].target = self.get_target(self.pop[idx].solution)
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(self.pop)
        fitness = np.array([agent.target.fitness for agent in self.pop])
        self.leaders_idx, self.winners_idx = self.get_leaders_and_winners(fitness)

        # Parliamentary Affairs Phase
        for j in range(self.pop_n):
            c_j_idx = self.winners_idx[j]
            # Select a random constituency winner r (where r != j)
            r = self.sample_indexes_exclude_one(self.generator, self.pop_n, exclude_idx=j, n_samples=1)
            c_r_idx = self.winners_idx[r]

            a = self.generator.random()
            c_new = self.pop[c_r_idx].solution + (2 * a - 1) * np.abs(self.pop[c_r_idx].solution - self.pop[c_j_idx].solution)
            pos_new = self.correct_solution(c_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[c_j_idx].target, self.problem.minmax):
                self.pop[c_j_idx] = agent

        # Update historical archives and party switching rate
        self.pop_prev = pop_temp
        self.lam -= self.lamda_max / self.epoch
