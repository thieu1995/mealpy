# !/usr/bin/env python
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseGA(Optimizer):
    """
    The original version of: Genetic Algorithm (GA)

    Links:
        1. https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        2. https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        3. https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation_multipoints (bool): Optional, True or False, effect on mutation process
        + mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.GA import BaseGA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> pc = 0.9
    >>> pm = 0.05
    >>> model1 = BaseGA(problem_dict1, epoch, pop_size, pc, pm)
    >>> best_position, best_fitness = model1.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    >>>
    >>> model2 = BaseGA(problem_dict1, epoch, pop_size, pc, pm, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = BaseGA(problem_dict1, epoch, pop_size, pc, pm, crossover="one_point", mutation="scramble")
    >>>
    >>> model4 = BaseGA(problem_dict1, epoch, pop_size, pc, pm, crossover="arithmetic", mutation_multipoints=True, mutation="swap")
    >>>
    >>> model5 = BaseGA(problem_dict1, epoch, pop_size, pc, pm, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = BaseGA(problem_dict1, epoch, pop_size, pc, pm, selection="random", mutation="inversion")
    >>>
    >>> model7 = BaseGA(problem_dict1, epoch, pop_size, pc, pm, crossover="arithmetic", mutation="flip")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pc=0.95, pm=0.025, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pc (float): cross-over probability, default = 0.95
            pm (float): mutation probability, default = 0.025
            selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            k_way (float): Optional, set it when use "tournament" selection, default = 0.2
            crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = False
            mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point, default="flip"
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm

        self.selection = "tournament"     # "random", "roulette"
        self.k_way = 0.2
        self.crossover = "uniform"    # "one_point", "multi_points", "uniform", "arithmetic"
        self.mutation = "flip"          # "swap", "scramble", "inversion"
        self.mutation_multipoints = False

    def selection_process(self, list_fitness):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        + Default selection strategy is Tournament with k% = 0.2.
        + Other strategy like "roulette" and "random" can be selected via Optional parameter "selection"

        Args:
            list_fitness (np.array): list of fitness values.

        Returns:
            list: The position of dad and mom
        """
        if self.selection == "roulette":
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
        elif self.selection == "random":
            id_c1, id_c2 = np.random.choice(range(self.pop_size), 2, replace=False)
        else:   ## tournament
            id_c1, id_c2 = self.get_index_kway_tournament_selection(self.pop, k_way=self.k_way, output=2)
        return self.pop[id_c1][self.ID_POS], self.pop[id_c2][self.ID_POS]

    def crossover_process(self, dad, mom):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
        + Default crossover strategy is "uniform"
        + Other strategy like "arithmetic", "one_point", "multi_points" can be selected via parameter: crossover

        Args:
            dad (np.array): The position of dad
            mom (np.array): The position of mom

        Returns:
            list: The position of child 1 and child 2
        """
        if self.crossover == "arithmetic":
            w1, w2 = self.crossover_arthmetic_recombination(dad, mom)
        elif self.crossover == "one_point":
            cut = np.random.randint(1, self.problem.n_dims-1)
            w1 = np.concatenate([ dad[:cut], mom[cut:] ])
            w2 = np.concatenate([ mom[:cut], dad[cut:] ])
        elif self.crossover == "multi_points":
            idxs = np.random.choice(range(1, self.problem.n_dims-1), 2, replace=False)
            cut1, cut2 = np.min(idxs), np.max(idxs)
            w1 = np.concatenate([ dad[:cut1], mom[cut1:cut2], dad[cut2:] ])
            w2 = np.concatenate([ mom[:cut1], dad[cut1:cut2], mom[cut2:] ])
        else:           # uniform
            flip = np.random.randint(0, 2, self.problem.n_dims)
            w1 = dad * flip + mom * (1 - flip)
            w2 = mom * flip + dad * (1 - flip)
        return w1, w2

    def mutation_process(self, child):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
        + There are 2 strategies that effects by the mutation probability: Mutated on single point or the whole vector.
            + Multiple points (whole vector) has 2 strategies selected via parameter: mutation
                + flip --> should set the pm small such as: [0.01 -> 0.2]
                + swap --> (default in this case) should set the pm small such as: [0.01 -> 0.2]
            + Single point has 4 strategies:
                + flip --> should set the pm large such as: [0.5 -> 0.9]
                + swap --> same as flip: pm in range [0.5 -> 0.9]
                + scramble --> should set the pm small enough such as: [0.4 -> 0.6]
                + inversion --> like scramble [0.4 -> 0.6]

        Args:
            child (np.array): The position of the child

        Returns:
            np.array: The mutated vector of the child
        """

        if self.mutation_multipoints:
            if self.mutation == "flip":
                mutation_child = np.random.uniform(self.problem.lb, self.problem.ub)
                flag_child = np.random.uniform(0, 1, self.problem.n_dims) < self.pm
                return np.where(flag_child, mutation_child, child)
            else:       # "swap"
                for idx in range(self.problem.n_dims):
                    idx_swap = np.random.choice(list(set(range(0, self.problem.n_dims)) - {idx}))
                    child[idx], child[idx_swap] = child[idx_swap], child[idx]
                    return child
        else:
            if self.mutation == "swap":
                idx1, idx2 = np.random.choice(range(0, self.problem.n_dims), 2, replace=False)
                child[idx1], child[idx2] = child[idx2], child[idx1]
                return child
            elif self.mutation == "inversion":
                cut1, cut2 = np.random.choice(range(0, self.problem.n_dims), 2, replace=False)
                temp = child[cut1:cut2]
                temp = temp[::-1]
                child[cut1:cut2] = temp
                return child
            elif self.mutation == "scramble":
                cut1, cut2 = np.random.choice(range(0, self.problem.n_dims), 2, replace=False)
                temp = child[cut1:cut2]
                np.random.shuffle(temp)
                child[cut1:cut2] = temp
                return child
            else:   # "flip"
                idx = np.random.randint(0, self.problem.n_dims)
                child[idx] = np.random.uniform(self.problem.lb[idx], self.problem.ub[idx])
                return child

    def survivor_process(self, pop, pop_child):
        """
        The current survivor process is select the worst solution out of k-way solutions (tournament selection) and
        compare with child solutions. The better solution will be kept for the next generation.

        Args:
            pop: The old population
            pop_child: The new population

        Returns:
            The new population
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            id_child = self.get_index_kway_tournament_selection(pop, k_way=0.1, output=1, reverse=True)[0]
            if self.compare_agent(pop_child[idx], pop[id_child]):
                pop_new.append(pop_child[idx])
            else:
                pop_new.append(pop[id_child])
        return pop_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_fitness = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
        pop_new = []
        for i in range(0, int(self.pop_size/2)):
            ### Selection
            child1, child2 = self.selection_process(list_fitness)

            ### Crossover
            if np.random.uniform() < self.pc:
                child1, child2 = self.crossover_process(child1, child2)

            ### Mutation
            child1 = self.mutation_process(child1)
            child2 = self.mutation_process(child2)

            pop_new.append([child1, None])
            pop_new.append([child2, None])

        ### Survivor Selection
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.survivor_process(self.pop, pop_new)
