#!/usr/bin/env python
# Created by "Thieu" at 13:59, 24/06/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalCOA(Optimizer):
    """
    The original version of: Coyote Optimization Algorithm (COA)

    Links:
        1. https://ieeexplore.ieee.org/document/8477769
        2. https://github.com/jkpir/COA/blob/master/COA.py  (Old version Mealpy < 1.2.2)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_coyotes (int): [3, 15], number of coyotes per group, default=5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, COA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = COA.OriginalCOA(epoch=1000, pop_size=50, n_coyotes = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Pierezan, J. and Coelho, L.D.S., 2018, July. Coyote optimization algorithm: a new metaheuristic
    for global optimization problems. In 2018 IEEE congress on evolutionary computation (CEC) (pp. 1-8). IEEE.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_coyotes: int = 5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_coyotes (int): number of coyotes per group, default=5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_coyotes = self.validator.check_int("n_coyotes", n_coyotes, [2, int(self.pop_size / 2)])
        self.set_parameters(["epoch", "pop_size", "n_coyotes"])
        self.n_packs = int(pop_size / self.n_coyotes)
        self.sort_flag = False

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop_group = self.generate_group_population(self.pop, self.n_packs, self.n_coyotes)
        self.ps = 1. / self.problem.n_dims
        self.p_leave = 0.005 * (self.n_coyotes ** 2)  # Probability of leaving a pack

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        age = 1
        return Agent(solution=solution, age=age)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Execute the operations inside each pack
        for p in range(self.n_packs):
            # Get the coyotes that belong to each pack
            self.pop_group[p] = self.get_sorted_population(self.pop_group[p], self.problem.minmax)
            # Detect alphas according to the costs (Eq. 5)
            # Compute the social tendency of the pack (Eq. 6)
            tendency = np.mean([agent.solution for agent in self.pop_group[p]])

            #  Update coyotes' social condition
            pop_new = []
            for i in range(self.n_coyotes):
                rc1, rc2 = self.generator.choice(list(set(range(0, self.n_coyotes)) - {i}), 2, replace=False)
                # Try to update the social condition according to the alpha and the pack tendency(Eq. 12)
                pos_new = self.pop_group[p][i].solution + self.generator.random() * (self.pop_group[p][0].solution - self.pop_group[p][rc1].solution) + \
                          self.generator.random() * (tendency - self.pop_group[p][rc2].solution)
                # Keep the coyotes in the search space (optimization problem constraint)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                agent.age = self.pop_group[p][i].age
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            # Evaluate the new social condition (Eq. 13)
            pop_new = self.update_target_for_population(pop_new)
            # Adaptation (Eq. 14)
            self.pop_group[p] = self.greedy_selection_population(self.pop_group[p], pop_new, self.problem.minmax)

            # Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
            id_dad, id_mom = self.generator.choice(list(range(0, self.n_coyotes)), 2, replace=False)
            prob1 = (1. - self.ps) / 2.
            # Generate the pup considering intrinsic and extrinsic influence
            pup = np.where(self.generator.random(self.problem.n_dims) < prob1, self.pop_group[p][id_dad].solution, self.pop_group[p][id_mom].solution)
            # Eventual noise
            pos_new = self.generator.normal(0, 1) * pup
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            # Verify if the pup will survive
            packs = self.get_sorted_population(self.pop_group[p], self.problem.minmax)
            # Find index of element has fitness larger than new child. If existed an element like that, new child is good
            if self.compare_target(agent.target, packs[-1].target, self.problem.minmax):
                if self.problem.minmax == "min":
                    packs = sorted(packs, key=lambda agent: agent.age)
                else:
                    packs = sorted(packs, key=lambda agent: agent.age, reverse=True)
                # Replace worst element by new child, New born child with age = 0
                packs[-1] = agent
                self.pop_group[p] = [agent.copy() for agent in packs]

        # A coyote can leave a pack and enter in another pack (Eq. 4)
        if self.n_packs > 1:
            if self.generator.random() < self.p_leave:
                id_pack1, id_pack2 = self.generator.choice(list(range(0, self.n_packs)), 2, replace=False)
                id1, id2 = self.generator.choice(list(range(0, self.n_coyotes)), 2, replace=False)
                self.pop_group[id_pack1][id1], self.pop_group[id_pack2][id2] = self.pop_group[id_pack2][id2], self.pop_group[id_pack1][id1]

        # Update coyotes ages
        for id_pack in range(0, self.n_packs):
            for id_coy in range(0, self.n_coyotes):
                self.pop_group[id_pack][id_coy].age += 1
        self.pop = [agent for pack in self.pop_group for agent in pack]
