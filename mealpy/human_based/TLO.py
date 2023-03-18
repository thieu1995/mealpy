#!/usr/bin/env python
# Created by "Thieu" at 10:14, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from functools import reduce
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseTLO(Optimizer):
    """
    The developed version: Teaching Learning-based Optimization (TLO)

    Links:
       1. https://doi.org/10.5267/j.ijiec.2012.03.007

    Notes
    ~~~~~
    + Use numpy np.array to make operations faster
    + The global best solution is used

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.TLO import BaseTLO
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
    >>> model = BaseTLO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R. and Patel, V., 2012. An elitist teaching-learning-based optimization algorithm for solving
    complex constrained optimization problems. international journal of industrial engineering computations, 3(4), pp.535-560.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            ## Teaching Phrase
            TF = np.random.randint(1, 3)  # 1 or 2 (never 3)
            list_pos = np.array([item[self.ID_POS] for item in self.pop])
            DIFF_MEAN = np.random.rand(self.problem.n_dims) * (self.g_best[self.ID_POS] - TF * np.mean(list_pos, axis=0))
            temp = self.pop[idx][self.ID_POS] + DIFF_MEAN
            pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        pop_child = []
        for idx in range(0, self.pop_size):
            ## Learning Phrase
            temp = deepcopy(self.pop[idx][self.ID_POS]).astype(float)
            id_partner = np.random.choice(np.setxor1d(np.array(range(self.pop_size)), np.array([idx])))
            if self.compare_agent(self.pop[idx], self.pop[id_partner]):
                temp += np.random.rand(self.problem.n_dims) * (self.pop[idx][self.ID_POS] - self.pop[id_partner][self.ID_POS])
            else:
                temp += np.random.rand(self.problem.n_dims) * (self.pop[id_partner][self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)


class OriginalTLO(BaseTLO):
    """
    The original version of: Teaching Learning-based Optimization (TLO)

    Links:
       1. https://github.com/andaviaco/tblo

    Notes
    ~~~~~
    + Third loops are removed
    + This version is inspired from above link

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.TLO import OriginalTLO
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
    >>> model = OriginalTLO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R.V., Savsani, V.J. and Vakharia, D.P., 2011. Teaching–learning-based optimization: a novel method
    for constrained mechanical design optimization problems. Computer-aided design, 43(3), pp.303-315.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.support_parallel_modes = False
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            ## Teaching Phrase
            TF = np.random.randint(1, 3)  # 1 or 2 (never 3)
            #### Remove third loop here
            list_pos = np.array([item[self.ID_POS] for item in self.pop])
            pos_new = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                      (self.g_best[self.ID_POS] - TF * np.mean(list_pos, axis=0))
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, target], self.pop[idx]):
                self.pop[idx] = [pos_new, target]

            ## Learning Phrase
            id_partner = np.random.choice(np.setxor1d(np.array(range(self.pop_size)), np.array([idx])))

            #### Remove third loop here
            if self.compare_agent(self.pop[idx], self.pop[id_partner]):
                diff = self.pop[idx][self.ID_POS] - self.pop[id_partner][self.ID_POS]
            else:
                diff = self.pop[id_partner][self.ID_POS] - self.pop[idx][self.ID_POS]
            pos_new = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * diff
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            if self.compare_agent([pos_new, target], self.pop[idx]):
                self.pop[idx] = [pos_new, target]


class ImprovedTLO(BaseTLO):
    """
    The original version of: Improved Teaching-Learning-based Optimization (ImprovedTLO)

    Links:
       1. https://doi.org/10.1016/j.scient.2012.12.005

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_teachers (int): [3, 10], number of teachers in class, default=5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.TLO import ImprovedTLO
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
    >>> n_teachers = 5
    >>> model = ImprovedTLO(epoch, pop_size, n_teachers)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R.V. and Patel, V., 2013. An improved teaching-learning-based optimization algorithm
    for solving unconstrained optimization problems. Scientia Iranica, 20(3), pp.710-720.
    """

    def __init__(self, epoch=10000, pop_size=100, n_teachers=5, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_teachers (int): number of teachers in class
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.n_teachers = self.validator.check_int("n_teachers", n_teachers, [2, int(np.sqrt(self.pop_size)-1)])
        self.set_parameters(["epoch", "pop_size", "n_teachers"])

        self.n_students = self.pop_size - self.n_teachers
        self.n_students_in_team = int(self.n_students / self.n_teachers)
        self.sort_flag = False

    def initialization(self):
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)
        sorted_pop, self.g_best = self.get_global_best_solution(self.pop)
        self.teachers = deepcopy(sorted_pop[:self.n_teachers])
        sorted_pop = sorted_pop[self.n_teachers:]
        idx_list = np.random.permutation(range(0, self.n_students))
        self.teams = []
        for id_teacher in range(0, self.n_teachers):
            group = []
            for idx in range(0, self.n_students_in_team):
                start_index = id_teacher * self.n_students_in_team + idx
                group.append(sorted_pop[idx_list[start_index]])
            self.teams.append(group)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for id_teach, teacher in enumerate(self.teachers):
            team = self.teams[id_teach]
            list_pos = np.array([student[self.ID_POS] for student in self.teams[id_teach]])  # Step 7
            mean_team = np.mean(list_pos, axis=0)
            pop_new = []
            for id_stud, student in enumerate(team):
                if teacher[self.ID_TAR][self.ID_FIT] == 0:
                    TF = 1
                else:
                    TF = student[self.ID_TAR][self.ID_FIT] / teacher[self.ID_TAR][self.ID_FIT]
                diff_mean = np.random.rand() * (teacher[self.ID_POS] - TF * mean_team)  # Step 8

                id2 = np.random.choice(list(set(range(0, self.n_teachers)) - {id_teach}))
                if self.compare_agent(teacher, team[id2]):
                    pos_new = (student[self.ID_POS] + diff_mean) + np.random.rand() * (team[id2][self.ID_POS] - student[self.ID_POS])
                else:
                    pos_new = (student[self.ID_POS] + diff_mean) + np.random.rand() * (student[self.ID_POS] - team[id2][self.ID_POS])
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    pop_new[-1] = self.get_better_solution([pos_new, target], student)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_wrapper_population(pop_new)
                pop_new = self.greedy_selection_population(team, pop_new)
            self.teams[id_teach] = pop_new

        for id_teach, teacher in enumerate(self.teachers):
            ef = round(1 + np.random.rand())
            team = self.teams[id_teach]
            pop_new = []
            for id_stud, student in enumerate(team):
                id2 = np.random.choice(list(set(range(0, self.n_students_in_team)) - {id_stud}))
                if self.compare_agent(student, team[id2]):
                    pos_new = student[self.ID_POS] + np.random.rand() * (student[self.ID_POS] - team[id2][self.ID_POS]) + \
                              np.random.rand() * (teacher[self.ID_POS] - ef * team[id2][self.ID_POS])
                else:
                    pos_new = student[self.ID_POS] + np.random.rand() * (team[id2][self.ID_POS] - student[self.ID_POS]) + \
                              np.random.rand() * (teacher[self.ID_POS] - ef * student[self.ID_POS])
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    pop_new[-1] = self.get_better_solution([pos_new, target], student)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_wrapper_population(pop_new)
                pop_new = self.greedy_selection_population(team, pop_new)
            self.teams[id_teach] = pop_new

        for id_teach, teacher in enumerate(self.teachers):
            team = self.teams[id_teach] + [teacher]
            team, local_best = self.get_global_best_solution(team)
            self.teachers[id_teach] = local_best
            self.teams[id_teach] = team[1:]

        self.pop = self.teachers + reduce(lambda x, y: x + y, self.teams)
