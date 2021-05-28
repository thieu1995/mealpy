#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 22:08, 01/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import exp, where, all
from numpy.random import uniform, randint
from mealpy.root import Root


class BaseSA(Root):
    """
        The original version of: Simulated Annealing (SA)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 max_sub_iter=10, t0=1000, t1=1, move_count=5, mutation_rate=0.1,
                 mutation_step_size=0.1, mutation_step_size_damp=0.99, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.max_sub_iter = max_sub_iter    # Maximum Number of Sub-Iteration (within fixed temperature)
        self.t0 = t0                                            # Initial Temperature
        self.t1 = t1                                            # Final Temperature
        self.move_count = move_count                            # Move Count per Individual Solution
        self.mutation_rate = mutation_rate                      # Mutation Rate
        self.mutation_step_size = mutation_step_size            # Mutation Step Size
        self.mutation_step_size_damp = mutation_step_size_damp  # Mutation Step Size Damp

    def mutate(self, position, sigma):
        mu = self.mutation_rate
        # Select Mutating Variables
        pos_new = position + sigma * uniform(self.lb, self.ub, self.problem_size)
        pos_new = where(uniform(0, 1, self.problem_size) < mu, position, pos_new)

        if all(pos_new == position):     # Select at least one variable to mutate
            pos_new[randint(0, self.problem_size)] = uniform()
        return self.amend_position_faster(pos_new)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Initial Temperature
        t = self.t0                                             # Initial Temperature
        t_damp = (self.t1 / self.t0) ** (1.0 / self.epoch)      # Calculate Temperature Damp Rate
        sigma = self.mutation_step_size                         # Initial Value of Step Size

        for epoch in range(0, self.epoch):

            # Sub-Iterations
            for g in range(0, self.max_sub_iter):

                # Create new population
                pop_new = []
                for i in range(0, self.pop_size):
                    for j in range(0, self.move_count):
                        # Perform Mutation (Move)
                        pos_new = self.mutate(pop[i][self.ID_POS], sigma)
                        fit_new = self.get_fitness_position(pos_new)
                        pop_new.append([pos_new, fit_new])

                # Columnize and Sort Newly Created Population
                pop_new, g_best = self.update_sorted_population_and_global_best_solution(pop_new, self.ID_MIN_PROB, g_best)
                pop_new = pop_new[:self.pop_size]

                # Randomized Selection
                for i in range(0, self.pop_size):
                    # Check if new solution is better than current
                    if pop_new[i][self.ID_FIT] < pop[i][self.ID_FIT]:
                        pop[i] = pop_new[i]
                    else:
                        # Compute difference according to problem type
                        delta = abs(pop_new[i][self.ID_FIT] - pop[i][self.ID_FIT])
                        p = exp(-delta / t)     # Compute Acceptance Probability
                        if uniform() <= p:      # Accept / Reject
                            pop[i] = pop_new[i]
            # Update Temperature
            t = t_damp * t
            sigma = self.mutation_step_size_damp * sigma

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
